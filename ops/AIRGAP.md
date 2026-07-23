# Moving this into the air-gapped environment

The rule the whole design follows: **nothing is fetched at runtime.** Every model
weight, wheel and npm package is resolved at build time, at home, and the result
is carried over as container images. If a pod ever needs the internet to start,
that is a bug.

## What has to cross the gap

| Artifact | Size | How |
|---|---|---|
| `rag-api` image | ~400 MB | `docker save` |
| `rag-worker` image | ~400 MB | `docker save` |
| `rag-ocr` image | ~3 GB | `docker save` |
| `rag-web` image | ~200 MB | `docker save` |
| `opensearch` + `dashboards` | ~1.5 GB | already in your registry, most likely |
| `apache/tika:3.0.0.0-full` | ~1 GB | `docker save` |

Model weights ride **inside** the API, worker and OCR images. They are not a
separate transfer and there is no PVC to populate.

## Procedure

### 1. At home — fetch models and build

```bash
python ops/fetch_models.py
cd web && npm ci && cd ..

docker build -f services/api/Dockerfile    -t rag-api:0.1.0    .
docker build -f services/worker/Dockerfile -t rag-worker:0.1.0 .
docker build -f services/ocr/Dockerfile    -t rag-ocr:0.1.0    .
docker build -f web/Dockerfile -t rag-web:0.1.0 web/
```

The web image takes **no build arg**. It used to need `NEXT_PUBLIC_API_BASE_URL`
baked in, which tied one image to one environment; the UI now calls `/api/v1/*`
on its own origin and the web server proxies that to `RAG_API_URL`, read at
runtime from the ConfigMap. One image crosses the gap and runs in dev, test and
prod unchanged — which also means a wrong URL is `oc set env`, not another trip
across the gap with a rebuilt image.

### 2. Export

```bash
docker pull apache/tika:3.0.0.0-full
docker save -o rag-images.tar \
  rag-api:0.1.0 rag-worker:0.1.0 rag-ocr:0.1.0 rag-web:0.1.0 \
  apache/tika:3.0.0.0-full
sha256sum rag-images.tar > rag-images.tar.sha256
```

Verify the checksum on arrival before doing anything else. A truncated layer
produces failures that look like application bugs.

### 3. Inside — load and push

```bash
sha256sum -c rag-images.tar.sha256
docker load -i rag-images.tar

REG=image-registry.openshift-image-registry.svc:5000/rag
for img in rag-api rag-worker rag-ocr rag-web; do
  docker tag $img:0.1.0 $REG/$img:0.1.0
  docker push $REG/$img:0.1.0
done
```

### 4. Deploy

```bash
oc new-project rag
oc apply -f deploy/openshift/
oc create job --from=cronjob/rag-ingest rag-ingest-initial
```

## The base-image question

The Dockerfiles use `registry.access.redhat.com/ubi9/*`, which needs internet at
build time. If your work environment builds images internally instead of
importing them, override the base to whatever your internal registry mirrors:

```bash
docker build --build-arg BASE_IMAGE=registry.internal.corp/ubi9/python-312:latest ...
```

## Building wheels inside instead of importing images

If policy requires images be built inside the network, you need a wheelhouse:

```bash
# at home, on Linux matching the target — wheels are platform-specific
pip download -r services/api/requirements.txt -d wheelhouse/ \
  --platform manylinux2014_x86_64 --python-version 312 --only-binary=:all:
pip download -r services/ocr/requirements.txt -d wheelhouse-ocr/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --platform manylinux2014_x86_64 --python-version 312 --only-binary=:all:
```

Then inside, `pip install --no-index --find-links=wheelhouse/ -r requirements.txt`.

For npm: `npm ci` at home, then `tar czf node_modules.tgz node_modules`, or point
`.npmrc` at your internal Artifactory/Nexus mirror.

## Checklist before you carry anything over

- [ ] `docker compose up` works end to end at home
- [ ] `/api/v1/health/ready` returns ok with every component green
- [ ] A document ingests and is answerable with a correct page citation
- [ ] `RAG_LLM__PROVIDER` switching has been tested against a second endpoint
- [ ] The web image was built with the **work** API URL
- [ ] `deploy/models/SHA256SUMS` verifies
- [ ] Nothing in the images references `host.docker.internal` or `localhost`

## After the switch to the work LLM

Only these change:

```
RAG_LLM__PROVIDER=azure_openai
RAG_LLM__BASE_URL=https://<resource>.openai.azure.com
RAG_LLM__MODEL=<deployment-name>
RAG_LLM__API_VERSION=2024-10-21
RAG_LLM__API_KEY=<from the Secret>
```

Plus `RAG_LLM__CA_BUNDLE` if TLS is intercepted. Confirm with
`GET /api/v1/health/llm`, which reports the provider, model and base URL the pod
actually resolved.
