1. 
Prepare DataSet

2. 
Check cuda version:
  python
  import torch; torch.version.cuda
  exit()

3. 
  *if cuda 11.8*
  *if ampere or newer*
  pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
  *if older*
  pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"

  *if cuda 12.1*
  *if ampere or newer*
  pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
  *if older*
  pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

4. 
  git clone https://github.com/omar93939/vLLMandFineTune.git

5. 
  python vLLMandFineTune/tune.py