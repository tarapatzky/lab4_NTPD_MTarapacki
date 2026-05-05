# Laboratorium 04 - Konteneryzacja modelu ML
Projekt z przedmiotu: Nowoczesne Technologie Przetwarzania Danych.

## Opis zadania
Celem projektu było umieszczenie aplikacji API (FastAPI) wraz z modelem ML (klasyfikator Iris) w kontenerze Docker oraz konfiguracja środowiska wielokontenerowego z bazą Redis przy użyciu Docker Compose.

## Instrukcja uruchomienia

### 1. Lokalnie (bez Dockera)
1. Zainstaluj wymagane biblioteki: `pip install -r requirements.txt`
2. Uruchom serwer: `uvicorn app:app --host 0.0.0.0 --port 8000`

### 2. Za pomocą Dockera (pojedynczy kontener)
1. Zbuduj obraz: `docker build -t ml-api .`
2. Uruchom kontener: `docker run -p 8000:8000 ml-api`

### 3. Za pomocą Docker Compose (Zalecane)
1. Uruchom cały stos technologiczny (API + Redis): `docker-compose up --build`[cite: 3]

## Konfiguracja i zasoby
- **Porty**: API nasłuchuje na porcie 8000, Redis na 6379[cite: 3].
- **Zasoby**: Projekt wymaga środowiska Python 3.9-slim (określone w Dockerfile)[cite: 3].