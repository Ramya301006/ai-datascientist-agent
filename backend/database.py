from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)

DATABASE_URL = (
    "sqlite:///./chat_history.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)

SessionLocal = sessionmaker(
    bind=engine
)

Base = declarative_base()

class ChatHistory(Base):

    __tablename__ = "chat_history"

    id = Column(
        Integer,
        primary_key=True
    )

    question = Column(String)

    answer = Column(String)

Base.metadata.create_all(
    bind=engine , checkfirst=True
)