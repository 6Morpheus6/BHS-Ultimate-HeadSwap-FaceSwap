module.exports = {
  requires: {
    bundle: "ai",
  },
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          TF_ENABLE_ONEDNN_OPTS: "0"
         },
        path: "app",
        message: [
          "python main.py",
        ],
        on: [{
          "event": "/(http:\\/\\/[0-9.:]+)/",   
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
