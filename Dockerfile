FROM langchain/langgraph-api:3.11



# -- Adding local package . --
ADD . /deps/datasets
# -- End of local package . --

# -- Installing all local dependencies --
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*
# -- End of local dependencies install --
ENV LANGGRAPH_HTTP='{"port": 2024, "host": "0.0.0.0", "cors": {"allow_origins": ["*"], "allow_credentials": true, "allow_methods": ["*"], "allow_headers": ["*"]}}'
ENV LANGSERVE_GRAPHS='{"dataset_agent": "/deps/datasets/dataset_agent.py:app", "dataset_workflow": "/deps/datasets/dataset_agent.py:app"}'



# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license &&     touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing pip from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel &&     rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* &&     find /usr/local/bin -name "pip*" -delete
# -- End of pip removal --

WORKDIR /deps/datasets