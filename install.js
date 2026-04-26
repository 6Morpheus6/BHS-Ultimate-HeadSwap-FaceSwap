module.exports = {
  requires: {
    bundle: "ai",
  },
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
      when: "{{!exists('app/DeepFaceLab-master.zip')}}",
      method: "fs.download",
      params: {
        url: "https://github.com/iperov/DeepFaceLab/archive/refs/heads/master.zip",
        dir: "app"
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r ../requirements.txt"
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
      method: "hf.download",
      params: {
        "path":"app",
        "_": [ "Alissonerdx/BFS-Best-Face-Swap" ],
        "local-dir": "BFS-Best-Face-Swap"
      }
    },
    {
      method: "hf.download",
      params: {
        "path":"app",
        "_": [ "olesheva/head_swap_qwen_edit" ],
        "local-dir": "head_swap_qwen_edit"
      }
    },
    {
      method: "hf.download",
      params: {
        "_": [ "tonera/FLUX.2-klein-4B-fp8-diffusers" ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "../env",
        path: "app/VisoMaster",
        message: "python download_models.py"
      }
    }
  ]
}