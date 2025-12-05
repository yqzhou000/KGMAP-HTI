# -*- coding: utf-8 -*-
import os


class Config:
    def __init__(self):
        self.data_dir = "data"
        self.preprocessed_dir = "preprocessed"
        self.model_dir = "models"
        self.result_dir = "results"
        self.log_dir = "log"

        for dir_path in [self.preprocessed_dir, self.model_dir, self.result_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.num_relations = 7

        self.relation_names = [
            'herb-target',
            'herb-disease',
            'herb-ingredient',
            'ingredient-target',
            'target-disease',
            'herb-herb',
            'target-target'
        ]

        self.core_relations = [0, 2, 3]
        self.disease_relations = [1, 4]
        self.similarity_relations = [5, 6]

        self.use_weighted_similarity = True

        self.embedding_dim = 64
        self.transe_margin = 1.0
        self.transe_lr = 0.01
        self.transe_epochs = 100
        self.use_transe_init = True

        self.n_hid = 64
        self.n_steps = [3, 3, 3]
        self.dropout = 0.2
        self.attn_dim = 8

        self.use_path_enhancement = True
        self.path_lstm_hidden = 32
        self.path_attention_heads = 4

        self.learning_rate = 5e-3
        self.weight_decay = 1e-3
        self.batch_size = 512
        self.max_epochs = 100
        self.patience = 8
        
        self.num_folds = 5
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        self.gpu = 0
        self.seed = 42
        
        self.data_files = {
            'herb_target': 'herb_target.dat',
            'herb_disease': 'herb_disease.dat',
            'herb_ingredient': 'herb_ingredient.dat',
            'ingredient_target': 'ingredient_target.dat',
            'target_disease': 'target_disease.dat',
            'sim_herb': 'sim_herbs.dat',
            'herb_herb': 'herb_herb.dat',
            'sim_target': 'sim_targets.dat',
            'target_target': 'target_target.dat',
        }

        self.verbose = True
        self.log_interval = 10       
        
    def __repr__(self):
        info = []
        info.append("="*60)
        info.append("KGMAP-HTI Configuration (7 Relations - Correct)")
        info.append("="*60)
        info.append(f"Relations: {self.num_relations} types")
        info.append(f"  Core: {self.core_relations}")
        info.append(f"  Disease: {self.disease_relations}")
        info.append(f"  Similarity: {self.similarity_relations}")
        info.append(f"Weighted Similarity: {self.use_weighted_similarity}")
        info.append(f"TransE Init: {self.use_transe_init}")
        info.append(f"Model: n_hid={self.n_hid}, n_steps={self.n_steps}")
        info.append(f"Training: lr={self.learning_rate}, batch={self.batch_size}")
        info.append(f"Cross-validation: {self.num_folds} folds")
        info.append("="*60)
        return "\n".join(info)


if __name__ == "__main__":
    config = Config()
    print(config)

    assert config.num_relations == 7
    assert len(config.relation_names) == 7
    print("\n✓ Configuration is correct!")
