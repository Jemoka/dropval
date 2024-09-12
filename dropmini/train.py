from synthetic import Trainer

base = Trainer("models/no_dropout", 1, 16, dropout_pct=0)
base.train()

for dropout_amount in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    alt = Trainer(f"models/dropout_{dropout_amount}", 1, 16,
                  dropout_pct=dropout_amount)
    alt.train_dl = base.train_dl
    alt.dev_dl = base.dev_dl
    alt.train()
