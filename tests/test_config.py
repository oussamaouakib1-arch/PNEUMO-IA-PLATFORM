from pneumonia.config import Settings


def test_default_settings_are_local() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.postgres_host == "localhost"
    assert settings.minio_bucket == "pneumonia"


def test_database_url_uses_configured_values() -> None:
    settings = Settings(
        _env_file=None,
        postgres_user="user",
        postgres_password="secret",
        postgres_host="database",
        postgres_port=5433,
        postgres_db="medical",
    )

    assert settings.database_url == "postgresql://user:secret@database:5433/medical"
