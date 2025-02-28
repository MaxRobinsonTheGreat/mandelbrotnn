import torch
import torch.nn as nn
import src.models as models
from train_image import trainImage
from evolve_image import evolveImage

create_model = lambda: models.SkipConn(hidden_size=50, num_hidden_layers=5, activation=nn.GELU).cuda()
# create_model = lambda: models.Simple(hidden_size=50, num_hidden_layers=5, activation=nn.GELU).cuda()
# create_model = lambda: models.Fourier(16, hidden_size=50, num_hidden_layers=5, linmap=models.CenteredLinearMap(-1, 1, -1, 1, 2*torch.pi, 2*torch.pi)).cuda()

print(sum(p.numel() for p in create_model().parameters()))

# path = 'DatasetImages/smiley.png'
path = 'DatasetImages/darwin_grey.jpg'
name = 'darwin'

# path = 'DatasetImages/shells.jpg'
# name = 'shells_fourier'

color_map = 'plasma'

# EA (local search)
evolveImage(
    image_path=path,
    proj_name=name + '_evolve',
    create_model=create_model,
    lr=1,
    lr_decay=0.9,
    reset_lr_at=0.001,
    num_rounds=1000,
    pop_size=1000,
    init_rand_searches=1000,
    plt_color_map=color_map,
)

# SGD
trainImage(
    image_path=path,
    proj_name=name + '_sgd',
    model=create_model(),
    lr=0.8,
    batch_size=8000,
    num_epochs=200,
    optimizer='sgd',
    scheduler_step=30,
    scheduler_gamma=0.8,
    save_every_n_batches=10,
    plt_color_map=color_map,
)

# Adam
trainImage(
    image_path=path,
    proj_name=name + '_adam',
    model=create_model(),
    lr=0.02,
    batch_size=8000,
    num_epochs=200,
    optimizer='adam',
    scheduler_step=10,
    scheduler_gamma=0.5,
    save_every_n_batches=10,
    plt_color_map=color_map,
)