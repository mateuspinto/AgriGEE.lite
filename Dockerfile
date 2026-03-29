FROM ghcr.io/prefix-dev/pixi:latest AS build

WORKDIR /app
COPY pyproject.toml pixi.lock ./
RUN pixi install --frozen -e default

COPY . /app
RUN pixi install --frozen -e default

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY . /app
ENV PATH="/app/.pixi/envs/default/bin:$PATH"

CMD ["python", "agrigee_lite/foo.py"]
