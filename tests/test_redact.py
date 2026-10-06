import logging

from fplopt.redact import redact


def test_redact_masks_query_keys_and_bot_tokens():
    url = "https://x.test/v4/odds/?apiKey=abc123&regions=uk"
    assert redact(url) == "https://x.test/v4/odds/?apiKey=REDACTED&regions=uk"
    assert redact("https://api.telegram.org/bot123:AA-bb_c/sendMessage") == (
        "https://api.telegram.org/botREDACTED/sendMessage"
    )


def test_redact_masks_malformed_bot_tokens():
    masked = redact("POST https://api.telegram.org/bot12 3:x#y/sendMessage")
    assert masked == "POST https://api.telegram.org/botREDACTED/sendMessage"


def test_httpx_logger_output_is_redacted(caplog):
    caplog.set_level(logging.INFO, logger="httpx")
    logging.getLogger("httpx").info("HTTP Request: GET %s", "https://x.test/?apiKey=SECRET")
    assert "SECRET" not in caplog.text
    assert "apiKey=REDACTED" in caplog.text
