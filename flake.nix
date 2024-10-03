{
  description = "A basic dev flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
        with pkgs; {
          devShells.default = mkShell {
            buildInputs = [
              gcc
              gdb
              openmpi

              ffmpeg
              gnuplot
              clang-tools
              bc
            ];
          };

          LD_LIBRARY_PATH = lib.makeLibraryPath [
            openmpi
            clang-tools
          ];
        }
    );
}
