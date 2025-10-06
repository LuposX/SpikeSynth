{
  description = "SpikeSznth Dev Shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs";

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          # Include Python with your packages
          (pkgs.python3.withPackages (ps: with ps; [
            pandas
            requests
            numpy
            torch
            matplotlib
            jupyterlab
            polars
            tqdm
          ]))
        ];

        shellHook = ''
          echo "🐍 Python dev environment loaded!"
          python --version
        '';
      };
    };
}
