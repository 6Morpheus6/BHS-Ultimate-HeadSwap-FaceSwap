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
