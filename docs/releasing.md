# Releasing Kevala

Kevala's `ci` workflow validates a package artifact on `main`, and the `release` workflow publishes that exact validated tarball to npm through GitHub Actions OpenID Connect (OIDC) Trusted Publishing. Follow this guide to cut a release from a reviewed commit on `main`.

The commands use `RELEASE_VERSION`, `REVIEWED_COMMIT_SHA`, and `PACKAGE_TARBALL`. Set the version from `package.json`, and get the commit SHA and tarball path from its successful `ci` workflow run.

## Prepare and validate

1. Update the package, Rust workspace, and JavaScript runtime versions together. Regenerate `Cargo.lock` with `cargo generate-lockfile --offline`. The workspace has no external Rust dependencies.
2. Run the checks required by the change. The `ci` workflow runs Rust formatting, Clippy, Rust tests, WebAssembly builds and validation, CPU kernel guards, JavaScript syntax and tests, report integrity checks, and the package check.
3. Wait for the `ci` workflow on the exact `main` commit to pass. Its `package-<commit SHA>` artifact is created with `pnpm pack` (which runs the WebAssembly prepack build), checked with `pnpm check:package`, and retained for seven days. Download the artifact from that successful run and record the workflow run and commit SHA. Do not rebuild a different tarball for publication.
4. Check the downloaded tarball locally with the package validator and confirm the package version and contents before tagging:

   ```sh
   PACKAGE_TARBALL="path/to/kevala-${RELEASE_VERSION}.tgz"
   pnpm check:package "$PACKAGE_TARBALL"
   ```

The seven-day artifact retention window makes timely publication part of the release process.

## Tag the reviewed source

Use the exact commit that produced the validated artifact. Confirm that it is on `main` and that all required checks passed, then create and push a signed, annotated, version-prefixed tag:

```sh
git tag -s -a "v${RELEASE_VERSION}" "$REVIEWED_COMMIT_SHA" -m "kevala v${RELEASE_VERSION}"
git tag -v "v${RELEASE_VERSION}"
git show --no-patch --format='%H %D' "v${RELEASE_VERSION}"
git push origin "refs/tags/v${RELEASE_VERSION}"
git ls-remote origin "refs/tags/v${RELEASE_VERSION}^{}"
```

Set `REVIEWED_COMMIT_SHA` to the artifact's CI commit. The commit shown for the tag must match that SHA. Do not move an existing release tag to a different commit.

## Publish the validated package

Pushing `v${RELEASE_VERSION}` triggers `.github/workflows/release.yml`. You can also dispatch it manually for an existing signed tag on `main`:

```sh
gh workflow run release.yml --ref main -f tag="v${RELEASE_VERSION}" -f dist_tag=latest
```

The `release` workflow verifies that the tag is an annotated, GitHub-verified tag pointing to a commit on `main` whose `package.json` version matches `RELEASE_VERSION`. It downloads `package-${REVIEWED_COMMIT_SHA}` from the successful `ci` run for that commit, re-runs `scripts/check-package.mjs`, publishes the exact tarball to npm through OIDC Trusted Publishing with Sigstore provenance (no `NPM_TOKEN` secret), and checks the npm registry readback.

After the `release` workflow finishes, read back the registry metadata independently and compare it with the reviewed source and local tarball:

```sh
pnpm exec npm view "kevala@${RELEASE_VERSION}" version gitHead dist.integrity dist.tarball
printf 'sha512-'
openssl dgst -sha512 -binary "$PACKAGE_TARBALL" | openssl base64 -A
printf '\n'
```

Confirm `version` matches `RELEASE_VERSION` and `dist.integrity` matches the printed SHA-512 integrity of the downloaded artifact. Publishing a prebuilt tarball does not guarantee a `gitHead` field. If the registry reports one, it must identify the tagged commit. Otherwise, retain the successful commit-named CI artifact, Sigstore provenance attestation, and tarball digest as the source-to-package record. Investigate any conflicting commit or integrity before announcing the release.

## Publish the GitHub Release and verify Pages

Create or undraft the GitHub Release from the verified `v${RELEASE_VERSION}` tag only after npm readback succeeds. Describe the browser and Node package scope and link the release notes. Attach only artifacts that were built and validated for this release. Current CI does not build a native macOS CLI asset, so do not reuse an older binary as if it came from this commit.

Check the GitHub Release independently from npm. The Pages deployment is a separate `ci` job that runs for pushes to `main`, not for version tags. Verify its successful deployment and the served site separately, because package validation and npm publication do not prove that Pages has deployed the same source revision.
