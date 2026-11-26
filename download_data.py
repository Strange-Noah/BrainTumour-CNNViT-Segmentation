{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3f535f4f-f598-41d7-bdfb-c0e93aee2d23",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: http://mirrors.aliyun.com/pypi/simple\n",
      "Requirement already satisfied: nnunetv2 in ./miniconda3/lib/python3.10/site-packages (2.8.1)\n",
      "Requirement already satisfied: huggingface_hub in ./miniconda3/lib/python3.10/site-packages (2.4.0)\n",
      "Requirement already satisfied: pandas in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (2.3.3)\n",
      "Requirement already satisfied: nibabel in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (5.4.2)\n",
      "Requirement already satisfied: acvl-utils<0.3,>=0.2.6 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.2.7)\n",
      "Requirement already satisfied: requests in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (2.31.0)\n",
      "Requirement already satisfied: yacs in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.1.8)\n",
      "Requirement already satisfied: numpy>=1.24 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (1.26.3)\n",
      "Requirement already satisfied: scikit-image>=0.19.3 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.25.2)\n",
      "Requirement already satisfied: batchgeneratorsv2>=0.3.2 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.3.8)\n",
      "Requirement already satisfied: scipy in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (1.15.3)\n",
      "Requirement already satisfied: matplotlib in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (3.8.2)\n",
      "Requirement already satisfied: SimpleITK>=2.2.1 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (2.5.8)\n",
      "Requirement already satisfied: graphviz in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.21)\n",
      "Requirement already satisfied: tqdm in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (4.64.1)\n",
      "Requirement already satisfied: dynamic-network-architectures<0.5,>=0.4.4 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.4.4)\n",
      "Requirement already satisfied: seaborn in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.13.2)\n",
      "Requirement already satisfied: tifffile in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (2025.5.10)\n",
      "Requirement already satisfied: scikit-learn in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (1.7.2)\n",
      "Requirement already satisfied: batchgenerators>=0.25.1 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.25.6)\n",
      "Requirement already satisfied: blosc2>=3.3.2 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (4.3.3)\n",
      "Requirement already satisfied: einops in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (0.8.2)\n",
      "Requirement already satisfied: imagecodecs in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (2025.3.30)\n",
      "Requirement already satisfied: torch!=2.9.*,>=2.1.2 in ./miniconda3/lib/python3.10/site-packages (from nnunetv2) (2.1.2+cu121)\n",
      "Requirement already satisfied: httpx2<3,>=2.0.0 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (2.14.0)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (2023.12.2)\n",
      "Requirement already satisfied: packaging>=20.9 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (23.2)\n",
      "Requirement already satisfied: filelock>=3.10.0 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (3.13.1)\n",
      "Requirement already satisfied: hf-xet<2.0.0,>=1.6.0 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (1.8.0)\n",
      "Requirement already satisfied: tomli>=1.1.0 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (2.0.1)\n",
      "Requirement already satisfied: click<9.0.0,>=8.4.2 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (8.5.1)\n",
      "Requirement already satisfied: typing-extensions>=4.1.0 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (4.16.0)\n",
      "Requirement already satisfied: pyyaml>=5.1 in ./miniconda3/lib/python3.10/site-packages (from huggingface_hub) (6.0.1)\n",
      "Requirement already satisfied: connected-components-3d in ./miniconda3/lib/python3.10/site-packages (from acvl-utils<0.3,>=0.2.6->nnunetv2) (4.1.1)\n",
      "Requirement already satisfied: pillow>=7.1.2 in ./miniconda3/lib/python3.10/site-packages (from batchgenerators>=0.25.1->nnunetv2) (10.2.0)\n",
      "Requirement already satisfied: threadpoolctl in ./miniconda3/lib/python3.10/site-packages (from batchgenerators>=0.25.1->nnunetv2) (3.8.0)\n",
      "Requirement already satisfied: pydantic in ./miniconda3/lib/python3.10/site-packages (from blosc2>=3.3.2->nnunetv2) (2.14.4)\n",
      "Requirement already satisfied: msgpack in ./miniconda3/lib/python3.10/site-packages (from blosc2>=3.3.2->nnunetv2) (1.2.4)\n",
      "Requirement already satisfied: ndindex in ./miniconda3/lib/python3.10/site-packages (from blosc2>=3.3.2->nnunetv2) (1.10.1)\n",
      "Requirement already satisfied: numexpr>=2.14.1 in ./miniconda3/lib/python3.10/site-packages (from blosc2>=3.3.2->nnunetv2) (2.14.1)\n",
      "Requirement already satisfied: timm<1.0.23 in ./miniconda3/lib/python3.10/site-packages (from dynamic-network-architectures<0.5,>=0.4.4->nnunetv2) (1.0.22)\n",
      "Requirement already satisfied: truststore>=0.10 in ./miniconda3/lib/python3.10/site-packages (from httpx2<3,>=2.0.0->huggingface_hub) (0.10.5)\n",
      "Requirement already satisfied: idna>=3.18 in ./miniconda3/lib/python3.10/site-packages (from httpx2<3,>=2.0.0->huggingface_hub) (3.20)\n",
      "Requirement already satisfied: httpcore2==2.14.0 in ./miniconda3/lib/python3.10/site-packages (from httpx2<3,>=2.0.0->huggingface_hub) (2.14.0)\n",
      "Requirement already satisfied: anyio>=4.10 in ./miniconda3/lib/python3.10/site-packages (from httpx2<3,>=2.0.0->huggingface_hub) (4.16.0)\n",
      "Requirement already satisfied: h11>=0.16 in ./miniconda3/lib/python3.10/site-packages (from httpcore2==2.14.0->httpx2<3,>=2.0.0->huggingface_hub) (0.16.0)\n",
      "Requirement already satisfied: networkx>=3.0 in ./miniconda3/lib/python3.10/site-packages (from scikit-image>=0.19.3->nnunetv2) (3.2.1)\n",
      "Requirement already satisfied: lazy-loader>=0.4 in ./miniconda3/lib/python3.10/site-packages (from scikit-image>=0.19.3->nnunetv2) (0.6)\n",
      "Requirement already satisfied: imageio!=2.35.0,>=2.33 in ./miniconda3/lib/python3.10/site-packages (from scikit-image>=0.19.3->nnunetv2) (2.38.0)\n",
      "Requirement already satisfied: sympy in ./miniconda3/lib/python3.10/site-packages (from torch!=2.9.*,>=2.1.2->nnunetv2) (1.12)\n",
      "Requirement already satisfied: triton==2.1.0 in ./miniconda3/lib/python3.10/site-packages (from torch!=2.9.*,>=2.1.2->nnunetv2) (2.1.0)\n",
      "Requirement already satisfied: jinja2 in ./miniconda3/lib/python3.10/site-packages (from torch!=2.9.*,>=2.1.2->nnunetv2) (3.1.2)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in ./miniconda3/lib/python3.10/site-packages (from matplotlib->nnunetv2) (4.47.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in ./miniconda3/lib/python3.10/site-packages (from matplotlib->nnunetv2) (1.4.5)\n",
      "Requirement already satisfied: cycler>=0.10 in ./miniconda3/lib/python3.10/site-packages (from matplotlib->nnunetv2) (0.12.1)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in ./miniconda3/lib/python3.10/site-packages (from matplotlib->nnunetv2) (2.8.2)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in ./miniconda3/lib/python3.10/site-packages (from matplotlib->nnunetv2) (1.2.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in ./miniconda3/lib/python3.10/site-packages (from matplotlib->nnunetv2) (3.1.1)\n",
      "Requirement already satisfied: importlib-resources>=5.12 in ./miniconda3/lib/python3.10/site-packages (from nibabel->nnunetv2) (7.1.0)\n",
      "Requirement already satisfied: tzdata>=2022.7 in ./miniconda3/lib/python3.10/site-packages (from pandas->nnunetv2) (2026.6)\n",
      "Requirement already satisfied: pytz>=2020.1 in ./miniconda3/lib/python3.10/site-packages (from pandas->nnunetv2) (2026.6)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in ./miniconda3/lib/python3.10/site-packages (from requests->nnunetv2) (2022.12.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in ./miniconda3/lib/python3.10/site-packages (from requests->nnunetv2) (1.26.13)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in ./miniconda3/lib/python3.10/site-packages (from requests->nnunetv2) (2.0.4)\n",
      "Requirement already satisfied: joblib>=1.2.0 in ./miniconda3/lib/python3.10/site-packages (from scikit-learn->nnunetv2) (1.6.0)\n",
      "Requirement already satisfied: exceptiongroup>=1.0.2 in ./miniconda3/lib/python3.10/site-packages (from anyio>=4.10->httpx2<3,>=2.0.0->huggingface_hub) (1.2.0)\n",
      "Requirement already satisfied: cloudpickle>=3.0 in ./miniconda3/lib/python3.10/site-packages (from joblib>=1.2.0->scikit-learn->nnunetv2) (3.1.3)\n",
      "Requirement already satisfied: six>=1.5 in ./miniconda3/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib->nnunetv2) (1.16.0)\n",
      "Requirement already satisfied: torchvision in ./miniconda3/lib/python3.10/site-packages (from timm<1.0.23->dynamic-network-architectures<0.5,>=0.4.4->nnunetv2) (0.16.2+cu121)\n",
      "Requirement already satisfied: safetensors in ./miniconda3/lib/python3.10/site-packages (from timm<1.0.23->dynamic-network-architectures<0.5,>=0.4.4->nnunetv2) (0.8.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in ./miniconda3/lib/python3.10/site-packages (from jinja2->torch!=2.9.*,>=2.1.2->nnunetv2) (2.1.3)\n",
      "Requirement already satisfied: typing-inspection>=0.4.4 in ./miniconda3/lib/python3.10/site-packages (from pydantic->blosc2>=3.3.2->nnunetv2) (0.4.5)\n",
      "Requirement already satisfied: pydantic-core==2.50.5 in ./miniconda3/lib/python3.10/site-packages (from pydantic->blosc2>=3.3.2->nnunetv2) (2.50.5)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in ./miniconda3/lib/python3.10/site-packages (from pydantic->blosc2>=3.3.2->nnunetv2) (0.8.0)\n",
      "Requirement already satisfied: mpmath>=0.19 in ./miniconda3/lib/python3.10/site-packages (from sympy->torch!=2.9.*,>=2.1.2->nnunetv2) (1.3.0)\n",
      "\u001b[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv\u001b[0m\u001b[33m\n",
      "\u001b[0mNote: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install nnunetv2 huggingface_hub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "aae2ea72-2613-4149-9755-a755085f1a6e",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (1831996336.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;36m  Cell \u001b[0;32mIn[8], line 1\u001b[0;36m\u001b[0m\n\u001b[0;31m    huggingface-cli download --repo-type dataset Cupidqiu/UCSD-BMETS --local-dir /root/autodl-tmp/test_download\u001b[0m\n\u001b[0m                    ^\u001b[0m\n\u001b[0;31mSyntaxError\u001b[0m\u001b[0;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "huggingface-cli download --repo-type dataset Cupidqiu/UCSD-BMETS --local-dir /root/autodl-tmp/test_download"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fcd45ce0-76f5-431d-bb57-c1a5eab520f6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
