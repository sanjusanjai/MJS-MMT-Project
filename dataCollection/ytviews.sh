#!/bin/bash
# usage: https://www.youtube.com/watch?v=3pSPkYpS37c

curl -s $1 | grep viewCount | head -1 | perl -pe 's|.*(viewCount".*?,).*|\1|' | cut -d'"' -f 3
