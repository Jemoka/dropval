from synthetic import *

trainer = Trainer("models/dropout", 1, 16, dropout_pct=0.1)
trainer.train()

trainer2 = Trainer("models/no_dropout", 1, 16, dropout_pct=0)
trainer2.train_dl = trainer.train_dl
trainer2.dev_dl = trainer.dev_dl
trainer2.train()




