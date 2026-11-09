# Acurix Edge

Plataforma de inspeção visual local (edge) para linhas de produção. Captura, analisa e registra cada inspeção no próprio computador da estação, sem depender de internet, e torna visível qualquer falha em vez de aprovar por omissão.

> **Estado:** Foundation 0.1.0 — em desenvolvimento. Ainda não há modelo de visão aprovado; o sistema responde `NO_DECISION` até que um modelo e uma regra sejam aprovados. O software opera somente em modo de observação e não comanda máquinas.

## Princípios

- **Local-first:** captura, inferência, decisão e registro funcionam offline.
- **Falha explícita:** câmera, modelo, banco ou imagem fora do envelope nunca resultam em aprovação.
- **Rastreabilidade:** cada evento registra versão de software, configuração e modelo.
- **Somente observação:** nenhum comando para máquina, CLP ou atuador.

## Requisitos

- Python 3.12
- Windows 10/11 ou Linux
- Nenhuma dependência externa na Foundation 0.1.0 (somente biblioteca padrão)

## Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -e .
cp config/acurix.example.toml config/acurix.toml   # Windows: copy; ajuste conforme a estação
```

## Uso

```bash
acurix-edge init   --config config/acurix.toml   # cria ou migra o banco local
acurix-edge status --config config/acurix.toml   # estado e saúde em JSON (somente leitura)
acurix-edge reset-fault --reason "motivo" --config config/acurix.toml   # libera FAULT com auditoria
```

Códigos de saída de `status`: `0` OK, `1` degradado ou desconhecido, `2` falha ou configuração inválida.

## Testes

```bash
python -m unittest discover -s tests -t .
```

## Estrutura

```text
src/acurix_edge/   contratos, domínio (estados e decisão), configuração,
                   observabilidade, saúde, eventos, persistência e adaptadores
tests/             testes unitários e de contrato
config/            exemplo de configuração (sem segredos)
```

## Segurança

- A API e o console escutam somente em `127.0.0.1` por padrão.
- Segredos nunca são versionados; use `.env` local a partir de `.env.example`.
- Para relatar uma vulnerabilidade, contate o titular do projeto.

## Licença

Software proprietário. Copyright (c) 2026 [TITULAR A DEFINIR]. Todos os direitos reservados. Componentes de terceiros e seus avisos estão em `THIRD_PARTY_NOTICES`.
