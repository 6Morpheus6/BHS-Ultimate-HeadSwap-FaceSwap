module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/Hanzyusuf/BHS-HeadSwap.git app",
        ]
      },
      when: "{{!exists('app')}}"
    },
    {
      method: "shell.run",
      params: {
        message: [
          "git lfs install",
          "git clone https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap"
        ],
        path: "app"
      },
      when: "{{!exists('app/BFS-Best-Face-Swap')}}"
    },
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://huggingface.co/olesheva/head_swap_qwen_edit"
        ],
        path: "app"
      },
      when: "{{!exists('app/head_swap_qwen_edit')}}"
    },
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/HumanAIGC/SwapAnyHead.git"
        ],
        path: "app"
      },
      when: "{{!exists('app/SwapAnyHead')}}"
    },
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/visomaster/VisoMaster.git"
        ],
        path: "app"
      },
      when: "{{!exists('app/VisoMaster')}}"
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