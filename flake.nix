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

          # Compiled Python wheels (numpy, scipy, etc.) contain .so files that
          # depend on libstdc++ and libz.  In Nix these live in the store, not
          # at /usr/lib, so we must tell the dynamic linker where to find them.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            echo ""
            echo "  ╔═══════════════════════════════════════╗"
            echo "  ║      ScalpEdge Dev Shell Ready        ║"
            echo "  ╚═══════════════════════════════════════╝"
            echo ""
            echo "  Run:  uv sync && uv run python main.py"
            echo ""

            # Virtual environment managed by uv
            export UV_PROJECT_ENVIRONMENT=".venv"
            # Use the Nix-provided Python — never download a separate one
            export UV_PYTHON_DOWNLOADS=never
            export UV_PYTHON="${python}/bin/python3"
          '';
        };
      }
    );
}
