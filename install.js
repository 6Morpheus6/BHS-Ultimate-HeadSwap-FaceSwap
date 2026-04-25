module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/Hanzyusuf/BHS-HeadSwap.git app",
      },
    },
    {
      method: "shell.run",
      params: {
        path: "app/BFS-Best-Face-Swap",
        message: [
          "git lfs install",
          "git clone https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap",
          "git lfs pull"
        ]
      },
    },
    {
      method: "shell.run",
      params: {
        path: "app/head_swap_qwen_edit",
        message: [
          "git clone https://huggingface.co/olesheva/head_swap_qwen_edit",
          "git lfs pull"
        ]
      },
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git clone https://github.com/HumanAIGC/SwapAnyHead.git"
      },
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git clone https://github.com/visomaster/VisoMaster.git"
      },
    },
    {
      method: "fs.download",
      params: {
        url: "https://github.com/iperov/DeepFaceLab/archive/refs/heads/master.zip",
        dir: "app"
      },
      when: "{{!exists('app/DeepFaceLab-master.zip')}}"
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install tensorrt==10.6.0 tensorrt-cu12_libs==10.6.0 tensorrt-cu12_bindings==10.6.0 --extra-index-url https://pypi.nvidia.com"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        env: {
          HF_TOKEN: "{{envs.HF_TOKEN}}"
        },
        message: [
          "python download_flux.py"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "../../env",
        path: "app/VisoMaster",
        message: [
          "python download_models.py"
        ]
      }
    }
  ]
}