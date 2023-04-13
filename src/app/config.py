import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    model_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../model.joblib")
    )
    encoder_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../encoder.joblib")
    )
    lb_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../lb.joblib")
    )

    class Config:
        env_file = ".env"
