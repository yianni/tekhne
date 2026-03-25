# tekhne

[![CI](https://github.com/yianni/tekhne/actions/workflows/ci.yml/badge.svg)](https://github.com/yianni/tekhne/actions/workflows/ci.yml)

A tiny neural network in idiomatic Scala 3.

`tekhne` is a learning-first neural network project built from first principles in Scala 3. It focuses on immutable data, explicit math, and a clean structure that can grow into a reusable library.

## Why

- learn how neural networks work from first principles
- keep the implementation small and explicit
- use modern, idiomatic Scala 3
- build on a foundation that can evolve into a reusable library

## Project Structure

- `modules/core` - reusable neural network code
- `modules/demo` - runnable examples

## Current Scope

- dense feed-forward layers
- forward pass and backpropagation
- stochastic gradient descent training
- XOR and AND demo programs

## Non-goals

For now, `tekhne` is intentionally minimal. It is not trying to compete with production ML frameworks. The focus is correctness, clarity, and learning.

## Stack

- Scala 3.3.7
- sbt 1.12.5
- MUnit
- Scalafmt
- Scalafix
- scoverage

## Minimal Example

```scala
import scala.util.Random

import tekhne.*

val trainingData = Vector(
  (Vector(0.0, 0.0), Vector(0.0)),
  (Vector(0.0, 1.0), Vector(1.0)),
  (Vector(1.0, 0.0), Vector(1.0)),
  (Vector(1.0, 1.0), Vector(0.0))
)

val network = Network.random(
  layerSizes = Vector(2, 3, 1),
  activations = Vector(Activation.Tanh, Activation.Sigmoid),
  rng = new Random(42L)
)

val trained = Training.train(
  network,
  trainingData,
  TrainingConfig(
    learningRate = 0.1,
    epochs = 50_000
  )
)

val prediction = Forward.predict(trained, Vector(0.0, 1.0))
```

## Commands

```bash
sbt compile
sbt test
sbt "demo/runMain tekhne.demo.runTekhne"
sbt "demo/runMain tekhne.demo.runAndGateDemo"
sbt core/test
sbt scalafmtAll
sbt scalafixAll
sbt "project core" clean coverage test coverageReport
```

## Roadmap

- mini-batch training
- additional losses and optimizers
- training metrics and reporting
