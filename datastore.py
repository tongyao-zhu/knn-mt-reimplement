# class for the datastore
# some code is taken from tutorials of the FAISS library
import argparse

import faiss  # make faiss available
import torch
import os
import tqdm
import numpy as np
import time

np.random.seed(1234)


class DataStore:
    def __init__(self):
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True  # to avoid GPU memory issue
        resources = faiss.StandardGpuResources()
        d = 1024  # dimension of keys
        n_centroids = 4096  # number of clustering centroids
        code_size = 64
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, n_centroids, code_size, 8)
        index = faiss.index_cpu_to_gpu(resources, 0, index, co)
        index.nprobe = 32
        self.vocab_size = -1  # to be set later
        # self.index = faiss.IndexFlatL2(d)  # build the index
        self.index = index
        # print(self.index.is_trained)
        # print(self.index.ntotal)

        self.T = 10  # temperature as described in the paper

    def load(self, saved_dir):
        print(f"Loading trained index and token ids lookup from {saved_dir}")
        self.index = faiss.read_index(os.path.join(saved_dir, "index.trained"))
        self.token_lookup_tensor = torch.tensor(
            torch.load(os.path.join(saved_dir, "token_ids.pt"))
        )

    def train_index(self, key_store):
        """
        Training the FAISS index. We will perform sampling from the keys.
        :param key_store:
        :return:
        """
        print(f"Training Index, this might take a long time.")
        random_indices = np.random.choice(
            np.arange(len(key_store)),
            size=[min(1000000, len(key_store))],
            replace=False,
        )
        start = time.time()
        self.index.train(key_store[random_indices])
        print(f"Training takes {(time.time() - start)} seconds")
        self.index = faiss.index_gpu_to_cpu(self.index)  # put back to CPU

    def read_feature_files(self, feature_dir, percentage=100):
        value_files = list(
            filter(lambda x: x.endswith("values.pt"), os.listdir(feature_dir))
        )
        value_files = value_files[: int(len(value_files) * (percentage / 100.0))]
        key_store = []
        token_id_store = []
        start_time = time.time()
        for file_name in tqdm.tqdm(value_files, total=len(value_files)):
            file_id = file_name.split("_values.pt")[0]
            curr_keys = torch.load(os.path.join(feature_dir, f"{file_id}.pt"))
            curr_token_ids = torch.load(
                os.path.join(feature_dir, f"{file_id}_values.pt")
            )
            key_store += curr_keys.cpu()
            token_id_store += curr_token_ids.cpu()
        key_store = np.stack(key_store)
        token_id_store = np.stack(token_id_store)
        print(
            f"Successfully loaded {len(key_store)} keys and values, used {time.time() - start_time} seconds"
        )
        return key_store, token_id_store

    def read_features_and_train(self, feature_dir, output_dir, percentage=100):
        key_store, token_id_store = self.read_feature_files(
            feature_dir=feature_dir, percentage=percentage
        )
        self.token_lookup_tensor = torch.tensor(token_id_store)
        self.train_index(key_store)
        self.add_keys(key_store)
        self.save(output_dir)
        return

    def add_keys(self, keys_to_add):
        print("Start adding index")
        start_time = time.time()
        self.index.add(keys_to_add)  # add vectors to the index
        print(f"Adding index takes {time.time() - start_time} seconds")

    def save(self, output_dir):
        # write the trained index
        faiss.write_index(self.index, os.path.join(output_dir, "index.trained"))
        # save the index for token_ids
        torch.save(self.token_lookup_tensor, os.path.join(output_dir, "token_ids.pt"))
        print(
            f"Successfully saved the trained index (index.trained, and token_ids.pt) to {output_dir}"
        )

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size

    def search_k(self, query, k):
        """
        Search for the top K nearest neighbors, along with the distance
        :param k: top k
        :param query: should have shape (num_queries, dim)
        :return: scores: should have shape (num_queries, vocab_size), contains scores for each token for each entry
        """
        assert (
            self.vocab_size >= 1
        ), "Please set the vocab size first (using set_vocab_size method) before the search!"
        D, I = self.index.search(
            query, k
        )  # D, I will have shape (num_queries, k), containing the distance and the index
        # print(I.shape)
        # print("self.token_lookup_tensor", self.token_lookup_tensor.shape)

        actual_token_ids = self.token_lookup_tensor[torch.tensor(I)]  # (num_queries, k)
        scores = torch.zeros((query.shape[0], self.vocab_size))
        distance = torch.softmax(-torch.tensor(D) / self.T, dim=-1)
        scores = scores.scatter(
            1, actual_token_ids, distance, reduce="add"
        )  # will assign the scores to indices and add them
        return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate raw feature tensors for building the datastore"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="the directory of the generated raw features",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="the directory to save the trained index files",
    )
    parser.add_argument(
        "--sample_percentage",
        type=int,
        default=100,  # by default, use all available data
        help="The percentage to use for training",
    )
    args = parser.parse_args()

    return args


def read_and_train():
    if not (torch.cuda.is_available()):
        print("Warning: Not training on GPU can be very slow")

    args = parse_args()
    datastore = DataStore()
    datastore.read_features_and_train(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        percentage=args.sample_percentage,
    )


if __name__ == "__main__":
    read_and_train()

    # The following code is for testing purpose. You can ignore it during the actual run time.
    #
    # d = 64  # dimension
    # nb = 100000  # database size
    # nq = 10000  # nb of queries
    # np.random.seed(1234)  # make reproducible
    # xb = np.random.random((nb, d)).astype('float32')
    # xb[:, 0] += np.arange(nb) / 1000.
    # xq = np.random.random((nq, d)).astype('float32')
    # xq[:, 0] += np.arange(nq) / 1000.
    #
    # print(xb.shape)
    # print(xq.shape)
    # import faiss  # make faiss available
    #
    # index = faiss.IndexFlatL2(d)  # build the index
    # print(index.is_trained)
    # index.add(xb)  # add vectors to the index
    # print(index.ntotal)
    # k = 4  # we want to see 4 nearest neighbors
    # D, I = index.search(xb[:5], k)  # sanity check
    # print(I)
    # print(D)
    # D, I = index.search(xq, k)  # actual search
    # print(I[:5])  # neighbors of the 5 first queries
    # print(I[-5:])  # neighbors of the 5 last queries
    #

    # a = {1:2, 3:4, 5:6}
    # for key, value in zip(*a.items()):
    #     print(key, value)
    # print("number of GPUs", faiss.get_num_gpu())
    # if True:
    #     # if this fails, it means that the GPU version was not comp
    #     assert faiss.StandardGpuResources, \
    #         "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
    #     res = faiss.StandardGpuResources()
    #     dev_no = 0
    # print("No problem in initializing gpu")

    # print('Starting to initialise')
    # datastore = DataStore(config = {"saved_dir": "./saved_gen", "vocab_size":40032})
    # datastore.read_features_and_train(feature_dir = "./saved_gen", output_dir = "./datastore_1", percentage=1)
    # print("Finished initialise")
    # a = torch.tensor([[1,2,3],[0,1,2]])
    # b = torch.tensor([[0.2,0.4,0.5],[2,4,5]])
    # c = torch.zeros(2,5)
    #
    # print(c[a].shape)
    # # c[a]=b
    # print(c)
    # print(a.shape)
    # import torch
    #
    # input = torch.randn(2, 4)
    # print(input)
    # output = torch.zeros(2, 5)
    # index = torch.tensor([[3, 1, 0, 0], [1, 2, 0, 3]])
    # output = output.scatter(1, index, input, reduce="add")
    # print(output)
    #
    # a = torch.tensor(torch.arange(10000)) + 1
    #
    # print(a.shape)
    # index = torch.tensor([[3, 1, 0, 0], [1, 2, 0, 3]])
    # print(a[index].shape)
    # print(a[index])
    # T = 10
    # D = torch.tensor([[3,3,2,1], [3,4,5,1]])
    # D = D/T
    # D = torch.softmax(D, dim=-1)
    # print(D)
    #
    # query = torch.randn((7, 1024)).cpu().numpy()
    # scores = datastore.search_k(query, k=13)
    # print(scores.shape)

    # import numpy as np
    #
    # d = 64  # dimension
    # nb = 100000  # database size
    # nq = 10000  # nb of queries
    # np.random.seed(1234)  # make reproducible
    # xb = np.random.random((nb, d)).astype('float32')
    # xb[:, 0] += np.arange(nb) / 1000.
    # xq = np.random.random((nq, d)).astype('float32')
    # xq[:, 0] += np.arange(nq) / 1000.
    #
    # import faiss
    #
    # nlist = 100
    # m = 8
    # k = 4
    # quantizer = faiss.IndexFlatL2(d)  # this remains the same
    # index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    # # 8 specifies that each sub-vector is encoded as 8 bits
    # print('Starting training')
    # index.train(xb)
    # print("End training")
    # index.add(xb)
    # D, I = index.search(xb[:5], k)  # sanity check
    # print(I)
    # print(D)
    # index.nprobe = 10  # make comparable with experiment above
    # D, I = index.search(xq, k)  # search
    # print(I[-5:])
