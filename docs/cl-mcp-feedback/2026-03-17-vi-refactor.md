## VI refactor

- `lisp-edit-form` で同一ファイルに helper を順番に `insert_before` すると、既存の `defun` をアンカーにした小さな分割リファクタを安全に進めやすかった。
- `lisp-read-file` の raw view で置換後の全体構造をすぐ確認でき、追加した internal helper の並び確認に十分だった。
- `load-system` は警告件数だけ返すので、リファクタ後に新規 warning が増えたかを追いたい場面では詳細参照の導線があると便利。
- `run-tests` の targeted 実行と full 実行を続けて回せるのは速く、REPL での同値確認から検証までの往復が短かった。
