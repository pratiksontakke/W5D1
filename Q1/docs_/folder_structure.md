# Medical RAG Assistant - Project Structure Overview

## Project Organization

```
medical-rag-assistant/
├── backend/                        # Python FastAPI Backend Service
│   ├── app/                       # Main Application Package
│   │   ├── main.py               # FastAPI App & Startup
│   │   ├── api/                  # API Layer
│   │   │   ├── v1/              # API Version 1
│   │   │   │   ├── endpoints.py # API Routes
│   │   │   │   └── schemas.py   # Request/Response Models
│   │   │   └── deps.py          # Dependencies
│   │   ├── core/                # Core Application Logic
│   │   │   ├── config.py        # Configuration
│   │   │   └── security.py      # Security Features
│   │   ├── rag/                 # RAG Pipeline Components
│   │   │   ├── pipeline.py      # Main RAG Logic
│   │   │   ├── retriever.py     # Vector DB & Retrieval
│   │   │   ├── generator.py     # LLM Generation
│   │   │   └── evaluator.py     # RAGAS Validation
│   │   └── services/            # Additional Services
│   │       └── document_processor.py
│   ├── tests/                   # Backend Tests
│   └── Dockerfile.backend       # Backend Container
│
├── frontend/                     # React Frontend
│   ├── public/                  # Static Assets
│   ├── src/                     # Source Code
│   │   ├── api/                 # API Integration
│   │   ├── components/          # React Components
│   │   ├── hooks/              # Custom Hooks
│   │   ├── pages/              # Page Components
│   │   └── App.js              # Main App Component
│   ├── Dockerfile.frontend      # Frontend Container
│   └── package.json            # Dependencies
│
├── ingestion/                   # Data Ingestion
│   ├── run_ingestion.py        # Ingestion Script
│   └── source_data/            # Raw Data Storage
│
├── evaluation/                  # RAGAS Evaluation
│   ├── run_evaluation.py       # Evaluation Script
│   └── dataset.jsonl           # Test Dataset
│
└── docker-compose.yml          # Container Orchestration
```

## Component Details

### Backend (`/backend`)
- **Purpose**: Handles all server-side logic, RAG pipeline, and API endpoints
- **Key Components**:
  - `app/api/`: REST API implementation with versioning
  - `app/rag/`: Core RAG pipeline implementation
  - `app/core/`: Application configuration and security
  - `app/services/`: Helper services and utilities

### Frontend (`/frontend`)
- **Purpose**: User interface for medical professionals
- **Key Components**:
  - `src/components/`: Reusable UI components
  - `src/api/`: Backend API integration
  - `src/pages/`: Main application pages
  - `src/hooks/`: Custom React hooks

### Ingestion (`/ingestion`)
- **Purpose**: Data processing and vector database population
- **Key Components**:
  - `run_ingestion.py`: Main ingestion script
  - `source_data/`: Storage for medical documents

### Evaluation (`/evaluation`)
- **Purpose**: RAGAS metrics evaluation and monitoring
- **Key Components**:
  - `run_evaluation.py`: Evaluation pipeline
  - `dataset.jsonl`: Golden dataset for testing

## Key Features

1. **Modular Architecture**
   - Clear separation of concerns
   - Independent scaling of components
   - Easy maintenance and updates

2. **Production-Ready Setup**
   - Docker containerization
   - Environment configuration
   - Version control friendly

3. **Evaluation Framework**
   - RAGAS metrics integration
   - Automated testing
   - Quality monitoring

4. **Security Considerations**
   - API key validation
   - Environment variable management
   - Secure data handling

## Development Workflow

1. **Setup**
   - Clone repository
   - Configure environment variables
   - Install dependencies

2. **Data Ingestion**
   - Add medical documents to `ingestion/source_data/`
   - Run ingestion script
   - Verify vector database population

3. **Development**
   - Start services with Docker Compose
   - Develop features in respective directories
   - Run tests and evaluations

4. **Deployment**
   - Build Docker images
   - Deploy containers
   - Monitor performance

## Best Practices

1. **Version Control**
   - Use `.gitignore` for sensitive files
   - Maintain clean commit history
   - Follow branching strategy

2. **Configuration**
   - Use `.env` for secrets
   - Maintain `.env.example`
   - Document all configurations

3. **Testing**
   - Write unit tests
   - Perform RAGAS evaluations
   - Monitor metrics

4. **Documentation**
   - Update README.md
   - Document API endpoints
   - Maintain change logs 