# Releasing Kevala

Kevala's CI validates a package artifact, but it does not publish to npm or create a GitHub Release. Follow this guide for a release from a reviewed commit on `main`.

The commands use `RELEASE_VERSION`, `REVIEWED_COMMIT_SHA`, and `PACKAGE_TARBALL`. Set the version from `package.json`; get the commit SHA and tarball path from its successful CI run.

## Prepare and validate

1. Update the package, Rust workspace, and JavaScript runtime versions together. Regenerate `Cargo.lock` with `cargo generate-lockfile --offline`; the workspace has no external Rust dependencies.
2. Run the checks required by the change. The CI workflow runs Rust formatting, Clippy, Rust tests, WebAssembly builds and validation, CPU kernel guards, JavaScript syntax and tests, report integrity checks, and the package check.
3. Wait for the CI workflow on the exact `main` commit to pass. Its `package-<commit SHA>` artifact is created with `pnpm pack` (which runs the WebAssembly prepack build), checked with `pnpm check:package`, and retained for seven days. Download the artifact from that successful run and record the workflow run and commit SHA. Do not rebuild a different tarball for publication.
4. Check the downloaded tarball with the package validator. Confirm the package version and contents before proceeding:

   ```sh
   PACKAGE_TARBALL="path/to/kevala-${RELEASE_VERSION}.tgz"
   pnpm check:package "$PACKAGE_TARBALL"
   ```

The existing CI workflow does not run on tag pushes and does not have an npm publish step. The seven-day artifact expiry makes timely download part of the release process.

## Tag the reviewed source

Use the exact commit that produced the validated artifact. Confirm that it is the reviewed `main` head and that all required checks passed, then create a signed, annotated, version-prefixed tag:

```sh
git tag -s -a "v${RELEASE_VERSION}" "$REVIEWED_COMMIT_SHA" -m "kevala v${RELEASE_VERSION}"
git tag -v "v${RELEASE_VERSION}"
git show --no-patch --format='%H %D' "v${RELEASE_VERSION}"
git push origin "refs/tags/v${RELEASE_VERSION}"
git ls-remote origin "refs/tags/v${RELEASE_VERSION}^{}"
```

Set `REVIEWED_COMMIT_SHA` to the artifact's CI commit. The commit shown for the tag must match that SHA. Do not move an existing release tag to a different commit.

## Publish the validated package

The package is published manually from the downloaded CI tarball. First confirm that the intended npm publisher is authenticated with `npm whoami`. If npm returns an authentication error, stop and resolve access through the normal account process; do not put credentials in commands, files, or logs.

Publish the exact checked artifact:

```sh
npm publish "$PACKAGE_TARBALL" --access public
```

Read back the registry metadata and compare it with the reviewed source and local tarball:

```sh
npm view "kevala@${RELEASE_VERSION}" version gitHead dist.integrity dist.tarball
printf 'sha512-'
openssl dgst -sha512 -binary "$PACKAGE_TARBALL" | openssl base64 -A
printf '\n'
```

Confirm `version` matches `RELEASE_VERSION` and `dist.integrity` matches the printed SHA-512 integrity of the downloaded artifact. Publishing a prebuilt tarball does not guarantee a `gitHead` field. If the registry reports one, it must identify the tagged commit. Otherwise, retain the successful commit-named CI artifact and its digest as the source-to-package record; do not infer a commit from the version alone. Investigate any conflicting commit or integrity before announcing the release.

## Publish the GitHub Release and verify Pages

Create the GitHub Release from the verified `v${RELEASE_VERSION}` tag after npm readback. Describe the browser/Node package scope and link the release notes. Attach only artifacts that were built and validated for this release. Current CI does not build a native macOS CLI asset; do not reuse an older binary as if it came from this commit.

Check the GitHub Release independently from npm. The Pages deployment is a separate CI job that runs for pushes to `main`, not for version tags. Verify its successful deployment and the served site separately; a successful package validation or npm publication does not prove that Pages has deployed the same source revision.
