import unittest
import numpy as np
from datastore import DataStore
from tests.test_constants import *
import torch


class DataStoreTests(unittest.TestCase):
    def test_initialisation(self):
        """
        Test the initialisation of the datastore class
        :return:
        """
        datastore = DataStore()
        # initially vocab size should be -1
        self.assertEqual(datastore.vocab_size, -1, "Wrong vocab size")
        self.assertIsNotNone(
            datastore.index, "Datastore index is none during initialisation"
        )
        self.assertFalse(
            datastore.index.is_trained, "Datastore index is not correctly initialised"
        )

    def test_load_from_features(self):
        """
        Test the method to read raw feature files
        :return:
        """
        datastore = DataStore()
        key_store, token_id_store = datastore.read_feature_files(
            feature_dir=FEATURE_DIR, percentage=100
        )
        self.assertEqual(
            len(key_store),
            len(token_id_store),
            f"The length of keys {len(key_store)} and values {len(token_id_store)} are not consistent",
        )
        self.assertEqual(
            key_store.shape[-1],
            DIM_KEY,
            f"The dimension of the loaded key is not {DIM_KEY}",
        )

    def test_load_from_saved_datastore(self):
        """
        Test the method to load from saved datastore
        :return:
        """
        datastore = DataStore()
        datastore.load(DATASTORE_DIR)
        self.assertEqual(
            (datastore.index.ntotal),
            len(datastore.token_lookup_tensor),
            "The length of keys and values are not consistent",
        )
        self.assertTrue(datastore.index.is_trained, "The loaded index is not trained")

    def test_training_index(self):
        """
        Test the method to train the index with datastore
        :return:
        """
        len_keystore = LEN_KEYSTORE
        dim_keystore = DIM_KEY
        key_store = np.random.rand(len_keystore, dim_keystore)
        datastore = DataStore()
        datastore.train_index(key_store)
        self.assertTrue(datastore.index.is_trained, "The datastore is not trained")

    def test_adding_keys(self):
        """
        Test the method of adding in keys to the trained datastore
        :return:
        """
        len_keystore = LEN_KEYSTORE
        dim_keystore = DIM_KEY
        key_store = np.random.rand(len_keystore, dim_keystore)
        datastore = DataStore()
        datastore.train_index(key_store)
        datastore.add_keys(key_store)
        self.assertEqual(
            (datastore.index.ntotal), len_keystore, "Error in adding keys to the index"
        )  # ensure that the key are added

    def test_search(self):
        """
        Test the method of searching top k neighbors and return a KNN score over the vocab
        :return:
        """
        datastore = DataStore()
        datastore.set_vocab_size(vocab_size=VOCAB_SIZE)
        datastore.load(DATASTORE_DIR)
        query = torch.randn((NUM_QUERY, DIM_KEY)).cpu().numpy()
        scores = datastore.search_k(query, k=TOP_K)
        scores_sum = scores.sum(-1)
        self.assertTrue(
            torch.all(torch.isclose(scores_sum, torch.ones_like(scores_sum))),
            "The scores for each token don't add up to 1",
        )
        self.assertEqual(
            scores.size(0),
            NUM_QUERY,
            "Returned scores tensor doesn't match number of queries",
        )
        self.assertEqual(
            scores.size(1), VOCAB_SIZE, "Returned scores tensor have wrong vocab shape"
        )


if __name__ == "__main__":
    unittest.main()
