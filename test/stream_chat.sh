curl http://localhost:8080/v1/chat/completions/streaming -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d "{\"messages\": [{\"role\": \"user\", \"content\": \"こんにちは。"}]}"