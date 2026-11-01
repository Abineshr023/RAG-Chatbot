# Generative AI Dental Knowledge Assistant – RAG Chatbot

 📌 About the Project

The Generative AI Dental Knowledge Assistant is a Retrieval-Augmented Generation (RAG) based chatbot designed to provide context-aware answers from dental knowledge documents.

The system processes uploaded PDF documents, extracts and splits their content into smaller chunks, generates embeddings, and retrieves relevant information using semantic similarity. It uses Extractive Question Answering with an OpenRouter LLM fallback to generate relevant responses.

The application also provides separate General and Technical Knowledge Bases with role-based document access.

 🚀 Key Features

- **PDF Document Processing:** Extract text from uploaded dental PDF documents using PyMuPDF.
- **Text Chunking:** Split extracted document content into smaller chunks for efficient retrieval.
- **Text Embeddings:** Generate semantic embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- **Semantic Search:** Retrieve relevant document chunks based on user questions.
- **Question Answering:** Generate answers using Extractive QA with an OpenRouter LLM fallback.
- **Knowledge Base Management:** Maintain separate General and Technical knowledge bases.
- **PDF Upload and Replacement:** Upload PDF documents and automatically replace previous documents and re-index the knowledge base.
- **Role-Based Access Control (RBAC):** Manage access according to user roles.
- **REST API Integration:** Handle chatbot requests, authentication, and document operations through Django REST Framework.
- **React Chat Interface:** Provide an interactive conversational interface for users.

 🛠️ Technologies Used

 Backend
- Python
- Django
- Django REST Framework
- MySQL
- REST APIs

 AI / NLP
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Extractive Question Answering
- OpenRouter API
- PyMuPDF

 Frontend
- React.js
- JavaScript
- HTML
- CSS

 Authentication and Security
- Token Authentication
- Role-Based Access Control (RBAC)

 Tools
- Git
- GitHub
- PyCharm
- Postman

 🧠 RAG Pipeline Workflow

1. **PDF Upload:** Users upload dental knowledge documents into the appropriate knowledge base.
2. **Text Extraction:** PyMuPDF extracts text from the uploaded PDF.
3. **Text Chunking:** Extracted content is divided into smaller text chunks.
4. **Embedding Generation:** Sentence Transformers converts text chunks into vector embeddings.
5. **Semantic Retrieval:** The system compares the user's question with stored embeddings to retrieve relevant chunks.
6. **Answer Generation:** Extractive QA processes the retrieved context. OpenRouter provides an LLM fallback when required.
7. **Response:** The chatbot returns a context-aware answer to the user.

 👥 User Roles and Access Control

The application supports four user roles:

| User Role | Access |
|---|---|
| Superadmin | Access to General and Technical Knowledge Bases |
| Support Admin | Technical Knowledge Base |
| Product Admin | General Knowledge Base |
| General User | General Knowledge Base (read-only) |

 📚 Knowledge Base Structure

 General Knowledge Base
Contains general dental product and knowledge documents intended for general users and product-related queries.

 Technical Knowledge Base
Contains technical documentation intended for support and technical-related queries.

 ⚙️ Installation and Setup

 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd mlc1
```

 2. Create a Virtual Environment

```bash
python -m venv venv
```

 3. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

 4. Install Dependencies

```bash
pip install -r requirements.txt
```

 5. Configure Environment Variables

Create a `.env` file in the project root and configure the required database and API credentials.

Example:

```env
SECRET_KEY=your_django_secret_key
DEBUG=True
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=3306
OPENROUTER_API_KEY=your_openrouter_api_key
```

Use the exact environment variable names expected by your Django settings and application code.

 6. Apply Database Migrations

```bash
python manage.py migrate
```

 7. Run the Django Development Server

```bash
python manage.py runserver
```

 8. Start the React Frontend

Navigate to your frontend directory and run:

```bash
npm install
npm start
```

Use the frontend start command configured in your `package.json`.

 🎯 Project Objective

The main objective of this project is to build an AI-powered dental knowledge assistant that retrieves relevant information from domain-specific documents and provides context-aware answers through a conversational interface.

It combines document processing, semantic search, question answering, LLM integration, and role-based access control in a single application.

 👨‍💻 Developer

Abinesh R

Python | Django | React.js | RAG | NLP | MySQL

