.PHONY: sync dry-sync apply-sync

sync:
	./bin/file_window_sync --push

dry-sync:
	./bin/file_window_sync

apply-sync:
	./bin/file_window_sync --apply
