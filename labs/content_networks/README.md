# General

Content today is dispersed across fragmented platforms, and systems, some measured, most not.  Content here refers to movies, shows, books, podcasts, videos, reels, tweets, pictures, music and so on. Content as such can be seen as much higher level evolutionary abstractions emerging above  `memes`.  

```mermaid

classDiagram

    class Meme {
        + summary(): String
    }

    class MemeticExpression {
        explicit
        implicit
        ironic
        post_ironic
        meta_ironic
    }

    class Idea {
        - memes: List<Meme>
        - expressions: List<MemeticExpression>
        - strengths: List<Float>
        + __init__(memes: List<Meme>, expressions: List<MemeticExpression>, strengths: List<Float>)
    }

    class Content {
        + id(): String
        + name(): String
        + summary(): String
        + format(): String
        + ideas(): List<Idea>
    }

    Meme <|-- Idea
    MemeticExpression <|-- Idea
    Idea <|-- Content

```