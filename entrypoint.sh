#!/bin/bash
set -e
set -o pipefail

main() {
    git config --global url."https://".insteadOf git://
    git config --global url."$GITHUB_SERVER_URL/".insteadOf "git@github.com":

    # update git submodules (important if you have themes)
    git submodule update --init --recursive

    zola build

    cd public

    # if you want to add any commands do it here e.g. `touch .nojekyll`

    git init
    git config user.name "GitHub Actions"
    git config user.email "github-actions-bot@users.noreply.github.com"
    git add .

    git commit -m "Deploy ${GITHUB_REPOSITORY} to ${GITHUB_REPOSITORY}:gh-pages"
    git push --force "https://${GITHUB_ACTOR}:${TOKEN}@github.com/${GITHUB_REPOSITORY}.git" master:gh-pages
}

main "$@"
