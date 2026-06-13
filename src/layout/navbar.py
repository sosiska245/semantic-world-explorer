from dash import html


def navbar():
    return html.Div(
        [
            html.Img(src="/assets/logo.svg", className="swe-logo"),
            html.Div(
                [
                    html.P("Semantic World Explorer", className="swe-title"),
                    html.P(
                        "Type concepts. Watch the world recolor by meaning.",
                        className="swe-subtitle",
                    ),
                ]
            ),
        ],
        className="swe-navbar",
    )
