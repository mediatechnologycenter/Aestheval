from setuptools import setup
	
install_requires = [
	"ConfigArgParse==1.5.3",
	"huggingface-hub==0.6.0",
	"numpy ",
	"opencv-python",
	"pandas==1.4.2",
	"Pillow",
	"requests==2.27.1",
	"scikit-learn==1.0.2",
	"scipy==1.8.0",
	"timm @ git+https://github.com/rwightman/pytorch-image-models.git@95feb1da41c1fe95ce9634b83db343e08224a8c5",
	"tokenizers==0.12.1",
	"torch==1.11.0",
	"torchmetrics==0.8.2",
	"torchsort==0.1.9",
	"torchvision==0.12.0",
	"tqdm==4.64.0",
	"transformers==4.19.2",
	"typing_extensions",
	"urllib3",
	"wandb==0.12.16",
	"gdown",
	"praw",
	"pmaw",
	"python-dotenv",
]

setup(
	name="aestheval",
	install_requires=install_requires,
    packages=['aestheval'],
	version="0.1",
	scripts=[]
)