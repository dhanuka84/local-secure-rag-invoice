# **Building a Context-Aware AI System That Learns, Reuses, Retrieves, Decides, and Validates — for Automated Invoice Processing**

Automating invoice processing sounds simple — until you try to do it securely, reliably, and at scale.

In real business environments, invoices vary widely. A modern system needs to:

* Parse PDFs and image-only scans

* Detect vendor layout automatically

* Learn new invoice templates on the fly

* Validate extracted fields using a vision model

* Govern template promotion securely

* Run *entirely offline* in an air-gapped environment

This guide walks you through a production-grade architecture that achieves all of this using:

* **LangGraph** → context engineering & workflow automation

* **Milvus** → layout-similarity retrieval (RAG)

* **Redis** → template caching & success/failure metrics

* **Cerbos** → policy enforcement for template promotion

* **Ollama** → local LLM & vision self-evaluation

* **Custom Invoice Toolkit** → extraction, OCR, signature hashing


## **What Is Context Engineering?**

In modern LLM systems, **context engineering** means:

**Designing the flow, structure, and evolution of all the information that guides an LLM’s behavior — before, during, and after inference.**

It’s **not prompt engineering**.  
 It’s **not model fine-tuning**.  
 It’s the systematic orchestration of:

* state

* memory

* retrieval

* validation

* control flow

* policy

* learned templates

* environment variables

* role-based decisions

* vision inputs

* multi-step reasoning

It’s *workflow-level intelligence management*.

This pipeline demonstrates that very strongly.

Let’s break down how this AirGap AI system works end-to-end.

---

# **Part 1 — The Key Components**

This project is structured around several cooperating services, each with a clear responsibility.

---

## **Docker Compose — The Orchestration Layer**

 `docker-compose.yml` launches:

* **Redis** — caching \+ metrics

* **Milvus** — vector DB powering RAG

* **Ollama** — local LLM \+ vision model

* **Cerbos** — policy decision point

* Supporting infrastructure

All of this runs inside an **air-gapped network** — no external dependencies or cloud calls.

---

## **Redis — Fast Memory Layer for Templates & Metrics**

Redis serves two essential functions:

### **Template Cache**

Stores:

* active templates

* staging templates

* template promotion status

### **`template_cache.py`**

`def get_template(redis, signature):`  
    `key = f"template:{signature}:active"`  
    `tpl = redis.get(key)`  
    `if tpl:`  
        `return json.loads(tpl)`

    `staging_key = f"template:{signature}:staging"`  
    `tpl = redis.get(staging_key)`  
    `if tpl:`  
        `return json.loads(tpl)`

    `return None`

### **Metrics tracking**

Tracks:

* success counts

* failures

* vision\_pass / vision\_fail

* template usage counters

`def increment_success(redis, signature):`

    `redis.incr(f"metrics:{signature}:success_count")`

###  

This enables learning, promotion, and governance logic.

---

## **Milvus — Vector Search for Layout Retrieval**

Every invoice gets a **signature hash** derived from its textual structure.  
 This is embedded into a vector and stored in Milvus.

Milvus enables:

* layout similarity

* fast suggestions

* template reuse

This is Retrieval-Augmented Processing (RAP) for document pipelines.

### **`node_milvus_suggest` (from `nodes.py`)**

`results = milvus.search(collection="signatures", data=[state.embedding])`  
`if results and results[0].distance < 0.2:`  
    `state.suggested_signature = results[0].id`

---

## **Cerbos — Secure Policy Enforcement**

Cerbos externalizes business rules such as:

* Who can promote templates

* Roles allowed to bypass review

* Permission checks on actions

It decouples authorization from code.

### **`cerbos_client.py`**

`def can_promote_template(role: str) -> bool:`  
    `resp = cerbos_client.check(`  
        `principal={"id": "user", "roles": [role]},`  
        `resource={"kind": "template", "id": "promotion"},`  
        `actions=["promote"]`  
    `)`  
    `return resp.is_allowed("promote")`

---

## **LangGraph — The Workflow Conductor**

LangGraph provides:

* a deterministic state machine

* branching decisions

* node execution

* in-memory or Redis-checkpointed state persistence

* observability & debuggability

This is the **brain** coordinating all the specialists.

### **`build.py` (core of the pipeline)**

`with StateGraph(InvoiceState) as graph:`  
    `graph.add_node("extract_pdf", node_extract_pdf)`  
    `graph.add_node("ocr_if_needed", node_ocr_if_needed)`  
    `graph.add_node("signature", node_signature)`  
    `graph.add_node("check_cache", node_check_cache)`  
    `graph.add_conditional_edges("should_reuse_or_search", should_reuse_or_search)`  
    `graph.add_node("milvus_suggest", node_milvus_suggest)`  
    `graph.add_conditional_edges("should_use_suggest_or_learn", should_use_suggest_or_learn)`  
    `graph.add_node("learn_and_stage", node_learn_and_stage)`  
    `graph.add_node("extract_fields", node_extract_fields)`  
    `graph.add_node("vision_validate", node_vision_validate)`  
    `graph.add_conditional_edges("should_pass_or_review", should_pass_or_review)`  
    `graph.add_node("promote_template", node_promote_template)`  
    `graph.add_node("mark_for_review", node_mark_for_review)`  
    `graph.add_node("done", node_done)`

---

## **The Invoice Specialists — src/invoice/**

Each module is a “specialist”:

* **pdf\_io** → PDF extraction \+ OCR

* **signature** → signature hashing

* **template\_cache** → Redis template CRUD

* **template\_learner** → auto-regex rule generation via LLM

* **extract** → apply regex rules to text

* **vision\_validate** → vision model consistency check

* **cerbos\_client** → policy decisions

* **metrics** → record successes, failures, promotions

**This cleanly separates concerns between workflow (LangGraph) and operations (specialists).**

---

# **Part 2 — Architecture & Setup**

## **Dependencies (requirements.txt)**

Includes:

* langgraph

* langgraph-checkpoint

* redis

* pymilvus

* Cerbos client

* OCR & PDF processing libs

The presence of `langgraph-checkpoint>=1.0.0` plus a Redis instance means the system is fully ready for **persistent state workflows**.

### **requirements.txt**

`langgraph`  
`langgraph-checkpoint`  
`redis`  
`pymilvus`  
`cerbos`  
`ollama`  
`pdfplumber`  
`pytesseract`

---

## **Samples**

`samples/invoices/` contains:

* image-only invoices

* text-embedded PDFs

* various formats for testing

---

# **Part 3 — Workflow Structure: Conductor vs Specialists**

The project cleanly separates orchestration from behavior.

Flow Diagram:

[https://dhanuka84.blogspot.com/p/invoicesmallblogv6bigmindmap.html](https://dhanuka84.blogspot.com/p/invoicesmallblogv6bigmindmap.html)

---

## **The Conductor (src/graph\_invoice/)**

### **build.py**

Defines the entire workflow graph:

* every node

* conditional edges

* success/failure paths

This is the sheet music for the whole pipeline.

### **state.py**

Defines the `InvoiceState`, the object passed from node to node.

### **nodes.py**

Bridges workflow and specialists by:

* pulling in data

* calling specialists

* updating state

* returning results


### `state.py` — The InvoiceState Schema

`class InvoiceState(TypedDict):`  
    `pdf_path: str`  
    `text: str`  
    `images: List[Any]`  
    `signature: str`  
    `template: dict`  
    `extracted_fields: dict`  
    `vision_pass: bool`  
    `vision_score: float`  
    `role: str`  
    `done: bool`

---

### `nodes.py` — Node-to-Specialist Bridge

Example: PDF extraction node

`def node_extract_pdf(state: InvoiceState):`  
    `text, images = pdf_io.extract(state["pdf_path"])`  
    `state["text"] = text`  
    `state["images"] = images`  
    `return state`

Signature node

`def node_signature(state):`  
    `sig, vendor = signature.make_signature(state["text"])`  
    `state["signature"] = sig`  
    `state["vendor"] = vendor`  
    `return state`

Milvus suggestion node

`def node_milvus_suggest(state):`  
    `embedding = signature.embed(state["signature"])`  
    `state["embedding"] = embedding`  
    `return rag_suggest(state)`

Vision validation

`def node_vision_validate(state):`  
    `res = vision_validate.run(`  
        `fields=state["extracted_fields"],`  
        `images=state["images"]`  
    `)`  
    `state["vision_pass"] = res.pass_`  
    `state["vision_score"] = res.score`  
    `return state`

Template promotion

`def node_promote_template(state):`  
    `if not can_promote_template(state["role"]):`  
        `return state`  
    `template_cache.promote(`  
        `signature=state["signature"]`  
    `)`  
    `return state`

---

## **The Specialists (src/invoice/)**

Each file handles one business function — ingestion, extraction, learning, validation, caching, security.

---

# **Part 4 — The Graph in Motion (Step-by-Step)**

Let’s walk the journey of an invoice through the pipeline:

---

## **1\. Ingestion**

* Extract text & images using `pdf_io`

* Fall back to OCR if needed  
  text, images \= pdf\_io.extract(pdf\_path)  
  if len(text.strip()) \< 20:  
      text \= ocr.run(images)  
  

---

## **2\. Signature Identification**

`signature.py` computes:

* a structural signature hash

* vendor identification  
  signature \= sha256(text.encode()).hexdigest()\[:12\]

This allows layout recognition even for unseen formats.

---

## **3\. Cache Check**

Redis determines:

* Is there an active template?

* Is there a staging template?

If found → skip straight to extraction.

tpl \= template\_cache.get\_template(redis, signature)

---

## **4\. Decision 1: Reuse or Search?**

If no template exists → query Milvus.

---

## **5\. RAG Suggestion**

Milvus finds similar invoice signatures.

* If a similar invoice has a known template → reuse

* Otherwise → learn new template

results \= milvus.search(collection, vector)

---

## **6\. Decision 2: Suggest or Learn**

**use\_suggest**  
 → go directly to extraction

**learn**  
 → call the Template Learner module

---

## **7\. Template Learning**

The LLM learns:

* regex rules

* patterns for total/subtotal/tax

* numerical extraction

* structural hints

Template is saved as *staging*.

template \= llm.learn\_regex(text)  
template\_cache.save\_staging(signature, template)

---

## **8\. Extraction**

Regex rules produce:

* total

* tax

* subtotal

* invoice number

* line items

Math validation ensures consistency.

fields \= extract.apply(template, text)

---

## **9\. Vision Self-Evaluation**

A local vision model (via Ollama) verifies:

* Does the image actually show these values?

* Are totals readable?

* Do fields match the parsed text?

Returns:

* vision\_pass

* vision\_score

If score \< threshold → review  
 If score \>= threshold → promotion stage (if allowed)

res \= vision\_model.validate(fields, images)

---

## **10\. Decision 3: Pass or Review**

Self-evaluation determines next steps:

* success path → promotion

* failure path → manual review

All logged in Redis metrics.

---

## **11\. Template Promotion (Security Gate)**

Promotion requires:

### **1\. AUTO\_PROMOTE\_THRESHOLD**

Redis tracks success\_count.

### **2\. Cerbos Authorization**

Only approved roles (manager, auditor, etc.) may promote.

If both checks pass → staging → active.

if metrics.success\_count(signature) \>= threshold:  
    if cerbos.can\_promote(role):  
        template\_cache.promote(signature)

---

## **12\. Completion**

Final JSON state is printed showing:

* chosen/learned template

* extracted fields

* validation results

* promotion status

---

# **Part 5 — Workflow State Management (In-Memory vs Redis Checkpoints)**

LangGraph supports two modes of state persistence — and this project is already configured for both.

---

## **1\. In-Memory State (How run\_invoice\_graph.py Executes)**

 script calls:

**`run_invoice_graph.py:`**

`result = graph.invoke({"pdf_path": pdf, "role": role})`  
`print(result)`

This mode:

* runs synchronously

* keeps state in memory

* returns final InvoiceState at the end

* is perfect for CLI or simple flow

This is the default for development and testing.

---

## **2\. Redis Checkpoints (Persistent, Resumable LangGraph Runs)**

from langgraph.checkpoint import RedisSaver

config \= {"checkpoint\_saver": RedisSaver(redis\_url)}  
app \= graph.compile(config)

Project contains:

* `langgraph-checkpoint>=1.0.0`

* A Redis instance in `docker-compose.yml`

* A stateful workflow design

This means this graph is ready for **checkpointing**.

### **What checkpointing gives you:**

* Resume after crashes

* Pause workflows mid-run

* Long-running invoice batches

* Distributed processing

* Full audit trails

* Human-in-the-loop resume points

LangGraph can serialize the **InvoiceState** into Redis at every step.

This is crucial for **air-gapped enterprise environments**, where:

* reliability

* auditability

* resumability

* distributed load

* strict governance

…are required.

---

# **Part 6 — Hands-On: Running the System**

## **Start services**

Clone the below GitHub Repository: [https://github.com/dhanuka84/local-secure-rag-invoice](https://github.com/dhanuka84/local-secure-rag-invoice)

\`\`\`bash  
docker-compose up \-d

$ docker-compose ps  
WARN\[0000\] /home/dhanuka84/research/local-secure-rag-invoice/docker-compose.yml: the attribute \`version\` is obsolete, it will be ignored, please remove it to avoid potential confusion   
NAME      IMAGE                          COMMAND                  SERVICE   CREATED      STATUS                  PORTS  
cerbos    ghcr.io/cerbos/cerbos:latest   "/cerbos server"         cerbos    2 days ago   Up 16 hours (healthy)   0.0.0.0:3592-\>3592/tcp, \[::\]:3592-\>3592/tcp, 3593/tcp  
milvus    milvusdb/milvus:v2.4.3         "/tini \-- milvus run…"   milvus    2 days ago   Up 2 days               0.0.0.0:9091-\>9091/tcp, \[::\]:9091-\>9091/tcp, 0.0.0.0:19530-\>19530/tcp, \[::\]:19530-\>19530/tcp  
ollama    ollama/ollama:latest           "/bin/ollama serve"      ollama    2 days ago   Up 2 days               0.0.0.0:11434-\>11434/tcp, \[::\]:11434-\>11434/tcp  
redis     redis:7-alpine                 "docker-entrypoint.s…"   redis     2 days ago   Up 2 days               0.0.0.0:6379-\>6379/tcp, \[::\]:6379-\>6379/tcp

make models

## **Install dependencies**

python \-m venv .venv && source .venv/bin/activate

pip install \--upgrade pip

pip install \-r requirements.txt

## **Process an invoice**

APP\_ROLE\=manager python \-m src.graph\_invoice.run\_invoice\_graph samples/invoices/invoice1.pdf

Environment variables:

* `AUTO_PROMOTE_THRESHOLD=1` → enable fast promotion for testing

* `APP_ROLE=manager` → allowed to promote templates

### **Failed Scenario**

$ APP\_ROLE=manager python \-m src.graph\_invoice.run\_invoice\_graph samples/invoices/invoice1.pdf

{  
 "pdf": "samples/invoices/invoice1.pdf",  
 "signature": "acme\_corporation\_123\_main\_street\_invoice\_72246d14",  
 "template\_source": "learned",  
 "promotion\_status": null,  
 "fields": {  
   "invoice\_no": "INV-1001",  
   "date": "2025-11-05",  
   "subtotal": "100.00",  
   "tax": "7.50",  
   "total": "107.50",  
   "tax\_rate": "0.0750"  
 },  
 "vision\_pass": false,  
 "vision\_score": 0.0,  
 "vision\_critique": "In the image provided, there are no visible numbers to compare against the extracted invoice amount fields.",  
 "done": true  
}

(.venv)local-secure-rag-invoice$ python \-m src.invoice.templates\_cli list  
Active:

Staging:  
  acme\_corporation\_123\_main\_street\_invoice\_72246d14

###  **Success Scenario**

(.venv) dhanuka84@dhanuka84:\~/research/local-secure-rag-invoice$ APP\_ROLE=manager python \-m src.graph\_invoice.run\_invoice\_graph samples/invoices/invoice1.pdf

{  
 "pdf": "samples/invoices/invoice1.pdf",  
 "signature": "acme\_corporation\_123\_main\_street\_invoice\_72246d14",  
 "template\_source": "active",  
 "promotion\_status": "pending\_success\_0",  
 "fields": {  
   "invoice\_no": "INV-1001",  
   "date": "2025-11-05",  
   "subtotal": "100.00",  
   "tax": "7.50",  
   "total": "107.50",  
   "tax\_rate": "0.0750"  
 },  
 "vision\_pass": true,  
 "vision\_score": 1.0,  
 "vision\_critique": "All extracted fields match the images exactly.",  
 "done": true  
}

$ python \-m src.invoice.templates\_cli list

Active:

  acme\_corporation\_123\_main\_street\_invoice\_72246d14

Staging:

### **Checking the Redis Cache**

$ docker exec \-it redis redis-cli

keys \*

1) "invoice:template:acme\_corporation\_123\_main\_street\_invoice\_72246d14"  
2) "invoice\_metrics:acme\_corporation\_123\_main\_street\_invoice\_72246d14"  
127.0.0.1:6379\> HGETALL invoice:template:acme\_corporation\_123\_main\_street\_invoice\_72246d14  
(error) WRONGTYPE Operation against a key holding the wrong kind of value  
127.0.0.1:6379\> HGETALL invoice\_metrics:acme\_corporation\_123\_main\_street\_invoice\_72246d14  
1) "vision\_failures"  
2) "1"  
3) "updated\_at"  
4) "1763327448"  
5) "promotions"  
6) "1"

---

# **Conclusion — A Modern, Secure, Self-Learning Document AI Pattern**

This architecture is a blueprint for modern document automation in secure environments.

It brings together:

* **LangGraph** → deterministic & resumable workflows

* **Redis** → lightning-fast memory layer

* **Milvus** → layout-aware RAG

* **Ollama** → local LLM \+ vision validation

* **Cerbos** → enterprise-grade policy enforcement

* **Auto Template Learning** → zero-shot adaptation

The result is a fully air-gapped, intelligent, self-evaluating pipeline for tax calculation and invoice extraction — capable of evolving safely while staying compliant with organizational rules.

