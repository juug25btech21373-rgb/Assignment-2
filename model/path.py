import os


class path_set_BERT:
    """Resolve project resources relative to this file, not the shell cwd."""

    def __init__(self, dataset):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.project_root = project_root
        self.path_data_dir = os.path.join(project_root, "data", dataset)

        if dataset == "twitter":
            self.path_txt_data_train = os.path.join(
                self.path_data_dir, "en_train.csv"
            )
            self.path_txt_data_test = os.path.join(
                self.path_data_dir, "en_test.csv"
            )
            self.path_img_data = os.path.join(
                self.path_data_dir, "images"
            ) + os.sep
        else:
            self.path_txt_data = os.path.join(
                self.path_data_dir, "pheme_final_fb.csv"
            )
            self.path_img_data = os.path.join(
                self.path_data_dir, "images"
            ) + os.sep

        self.path_bert = os.path.join(
            project_root, "bert-base-uncased"
        )
        self.VOCAB = "vocab.txt"

        # Colab-generated PPMI/SVD embeddings. These are not pretrained
        # Freebase TransE vectors from the original paper implementation.
        self.path_transe = os.path.join(
            project_root, "data_derived", "entity_cooccurrence", "entity2vec.bin"
        )
        self.path_dic = os.path.join(
            project_root, "data_derived", "entity_cooccurrence", "entity2id.txt"
        )
