{
  description = "ScalpEdge — Intraday 5-minute candle scalping backtester";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
      in {
        devShells.default = pkgs.mkShell {
          name = "scalpedge";

          buildInputs = [
            python
            pkgs.uv
            pkgs.git
          ];

          shellHook = ''
            echo ""
            echo "  ╔═══════════════════════════════════════╗"
            echo "  ║      ScalpEdge Dev Shell Ready        ║"
            echo "  ╚═══════════════════════════════════════╝"
            echo ""
            echo "  Run:  uv sync && uv run python main.py"
            echo ""

            # Create a local virtual environment managed by uv
            export UV_PROJECT_ENVIRONMENT=".venv"
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '';
        };
      }
    );
}
