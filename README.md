# tekhne

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

- dataset shuffling during training
- smarter initialization strategies
- additional losses and optimizers
