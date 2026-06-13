"""Pipeline stages. Each stage is independent: read artifact -> work -> write artifact.

A stage never imports another stage. The only shared code is `contracts/` (shapes)
and `io/` (read/write/config). See ARCHITECTURE.md §2, §7.
"""
