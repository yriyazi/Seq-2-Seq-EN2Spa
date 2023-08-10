import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
inference_mode      = config['inference_mode']
learning_rate       = config['learning_rate']
num_epochs          = config['num_epochs']
seed                = config['seed']
ckpt_save_freq      = config['ckpt_save_freq']

Language1      = config['Language1']
Language2      = config['Language2']
MAX_LENGTH      = config['MAX_LENGTH']

MAX_LENGTH_Lang1 = config['MAX_LENGTH_Lang1']
MAX_LENGTH_Lang2 = config['MAX_LENGTH_Lang2']


batch_train_size        = config['batch']['batch_train_size']
Validation_Set          = config['batch']['Validation_Set']
batch_validation_size   = config['batch']['batch_validation_size']
batch_test_size         = config['batch']['batch_test_size']

# Access dataset parameters
dataset_path        = config['dataset']['path']
train_split         = config['dataset']['train_split']
Validation_split    = config['dataset']['Validation_split']
test_split          = config['dataset']['test_split']

# Access model architecture parameters
model_name          = config['model']['name']
pretrained          = config['model']['pretrained']
Hidden_layer        = config['model']['Hidden_layer']

# Access optimizer parameters
optimizer_name      = config['optimizer']['name']
weight_decay        = config['optimizer']['weight_decay']
opt_betas           = config['optimizer']['betas']
# Access scheduler parameters
scheduler_name  = config['scheduler']['name']
start_factor    = config['scheduler']['start_factor']
end_factor      = config['scheduler']['end_factor']


# print("configuration hass been loaded!!! \n successfully")
# print(learning_rate)