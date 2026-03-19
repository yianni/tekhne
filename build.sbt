ThisBuild / scalaVersion := "3.3.7"
ThisBuild / organization := "io.github.yianni"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / semanticdbEnabled := true

ThisBuild / scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Wunused:all"
)

lazy val commonSettings = Seq(
  versionScheme := Some("early-semver"),
  licenses := Seq("MIT" -> url("https://opensource.org/licenses/MIT"))
)

lazy val root = (project in file("."))
  .settings(
    name := "tekhne",
    publish / skip := true
  )
  .aggregate(core, demo)

lazy val core = (project in file("modules/core"))
  .settings(commonSettings)
  .settings(
    name := "tekhne",
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.2.4" % Test
    ),
    testFrameworks += new TestFramework("munit.Framework")
  )

lazy val demo = (project in file("modules/demo"))
  .dependsOn(core)
  .settings(commonSettings)
  .settings(
    name := "tekhne-demo",
    publish / skip := true
  )
