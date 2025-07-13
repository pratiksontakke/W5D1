# Q2

## Backend Structure

```
backend/
├── rag/                     # RAG (Retrieval Augmented Generation) module
│   ├── main.py             # FastAPI entry point
│   ├── chroma_utils.py     # Chroma vector store utilities
│   ├── db_utils.py         # Database operations (e.g., chat history, metadata)
│   ├── langchain_utils.py  # LangChain RAG logic and configuration
│   ├── pydantic_models.py  # Pydantic models for request/response validation
│   ├── requirements.txt    # Python dependencies
│   ├── chroma_db/         # Directory for Chroma persistence
│   └── tests/             # Unit and integration tests
│
└── api/                    # FastAPI module
    ├── alembic/           # Database migrations
    ├── src/               # Source code
    │   ├── technical/     # Technical support module
    │   │   ├── router.py      # Technical support endpoints
    │   │   ├── schemas.py     # Pydantic models
    │   │   ├── models.py      # Database models
    │   │   ├── dependencies.py # Router dependencies
    │   │   ├── config.py      # Local configs
    │   │   ├── constants.py   # Module-specific constants
    │   │   ├── exceptions.py  # Module-specific errors
    │   │   ├── service.py     # Module-specific business logic
    │   │   └── utils.py       # Non-business logic functions
    │   ├── billing/       # Billing module
    │   │   ├── router.py      # Billing endpoints
    │   │   ├── schemas.py     # Pydantic models
    │   │   ├── models.py      # Database models
    │   │   ├── dependencies.py # Router dependencies
    │   │   ├── config.py      # Local configs
    │   │   ├── constants.py   # Module-specific constants
    │   │   ├── exceptions.py  # Module-specific errors
    │   │   ├── service.py     # Module-specific business logic
    │   │   └── utils.py       # Non-business logic functions
    │   ├── features/      # Feature requests module
    │   │   ├── router.py      # Feature request endpoints
    │   │   ├── schemas.py     # Pydantic models
    │   │   ├── models.py      # Database models
    │   │   ├── dependencies.py # Router dependencies
    │   │   ├── config.py      # Local configs
    │   │   ├── constants.py   # Module-specific constants
    │   │   ├── exceptions.py  # Module-specific errors
    │   │   ├── service.py     # Module-specific business logic
    │   │   └── utils.py       # Non-business logic functions
    │   ├── config.py     # Global configs
    │   ├── models.py     # Global database models
    │   ├── exceptions.py # Global exceptions
    │   ├── pagination.py # Global module e.g. pagination
    │   ├── database.py   # DB connection related stuff
    │   └── main.py       # FastAPI application entry point
    ├── tests/            # Test directory
    │   ├── technical/    # Technical support tests
    │   ├── billing/      # Billing tests
    │   └── features/     # Feature requests tests
    ├── templates/        # Template directory
    │   └── index.html    # Index template
    ├── requirements/     # Requirements directory
    │   ├── base.txt     # Base requirements
    │   ├── dev.txt      # Development requirements
    │   └── prod.txt     # Production requirements
    ├── .env             # Environment variables
    ├── .gitignore       # Git ignore file
    ├── logging.ini      # Logging configuration
    └── alembic.ini      # Alembic configuration
```
