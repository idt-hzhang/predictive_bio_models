from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_kickoff_presentation import (
    BLUE,
    GREEN,
    INK,
    MUTED,
    ORANGE,
    PAPER,
    RED,
    TEAL,
    WHITE,
    add_background,
    add_bullets,
    add_card,
    add_footer,
    add_image,
    add_textbox,
    add_title,
    set_fill,
    set_run,
)


PACKAGE_DIR = ROOT / "sgRNA_synthesizability"
SLIDES_MD = PACKAGE_DIR / ".github" / "slides.md"
OUTPUT_DIR = ROOT / "analysis" / "modeling_outputs"
FEATURE_PLOT_DIR = ROOT / "analysis" / "feature_target_plots"
OUTPUT = ROOT / "sgRNA_synthesizability_modeling_summary.pptx"


def require_inputs() -> None:
    required = [
        SLIDES_MD,
        OUTPUT_DIR / "presentation_summary.md",
        OUTPUT_DIR / "learning_curve.png",
        OUTPUT_DIR / "mainpeak_regression_observed_vs_predicted.png",
        FEATURE_PLOT_DIR / "token_lT_count.png",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing presentation input assets:\n{missing_text}")


def add_metric(slide, left, top, width, height, label, value, note=None, fill=RGBColor(238, 244, 235)):
    card = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    set_fill(card, fill, RGBColor(218, 211, 198))
    value_box = add_textbox(slide, left + Inches(0.14), top + Inches(0.15), width - Inches(0.28), Inches(0.38))
    paragraph = value_box.text_frame.paragraphs[0]
    paragraph.text = value
    paragraph.alignment = PP_ALIGN.CENTER
    set_run(paragraph.runs[0], size=20, color=INK, bold=True)
    label_box = add_textbox(slide, left + Inches(0.14), top + Inches(0.57), width - Inches(0.28), Inches(0.38))
    paragraph = label_box.text_frame.paragraphs[0]
    paragraph.text = label
    paragraph.alignment = PP_ALIGN.CENTER
    set_run(paragraph.runs[0], size=10, color=MUTED, bold=True)
    if note:
        note_box = add_textbox(slide, left + Inches(0.18), top + Inches(0.95), width - Inches(0.36), height - Inches(1.02))
        paragraph = note_box.text_frame.paragraphs[0]
        paragraph.text = note
        paragraph.alignment = PP_ALIGN.CENTER
        set_run(paragraph.runs[0], size=9, color=MUTED)


def add_result_row(slide, y, endpoint, model, performance, color):
    marker = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.72), y + Inches(0.08), Inches(0.12), Inches(0.52))
    set_fill(marker, color)
    endpoint_box = add_textbox(slide, Inches(0.95), y, Inches(2.45), Inches(0.72))
    paragraph = endpoint_box.text_frame.paragraphs[0]
    paragraph.text = endpoint
    set_run(paragraph.runs[0], size=13, color=INK, bold=True)
    model_box = add_textbox(slide, Inches(3.55), y, Inches(3.1), Inches(0.72))
    paragraph = model_box.text_frame.paragraphs[0]
    paragraph.text = model
    set_run(paragraph.runs[0], size=12, color=INK)
    perf_box = add_textbox(slide, Inches(6.95), y, Inches(5.65), Inches(0.72))
    paragraph = perf_box.text_frame.paragraphs[0]
    paragraph.text = performance
    set_run(paragraph.runs[0], size=12, color=INK)


def add_feature_chip(slide, text, left, top, width, color):
    chip = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, Inches(0.34))
    set_fill(chip, color)
    frame = chip.text_frame
    frame.clear()
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = PP_ALIGN.CENTER
    set_run(paragraph.runs[0], size=9, color=WHITE, bold=True)


def slide_problem(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    band = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, prs.slide_width, Inches(7.5))
    set_fill(band, RGBColor(22, 40, 50))
    ribbon = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, Inches(6.82), prs.slide_width, Inches(0.18))
    set_fill(ribbon, GREEN)
    title = add_textbox(slide, Inches(0.74), Inches(0.88), Inches(11.9), Inches(1.2))
    paragraph = title.text_frame.paragraphs[0]
    paragraph.text = "sgRNA Synthesizability Prediction"
    set_run(paragraph.runs[0], size=39, color=WHITE, bold=True)
    subtitle = add_textbox(slide, Inches(0.8), Inches(2.16), Inches(10.8), Inches(0.62))
    paragraph = subtitle.text_frame.paragraphs[0]
    paragraph.text = "Modeling sequence-derived risk signals for HPLC QC review prioritization"
    set_run(paragraph.runs[0], size=20, color=RGBColor(222, 232, 226))
    add_metric(slide, Inches(0.84), Inches(3.6), Inches(2.6), Inches(1.55), "binary-labeled rows", "2,646", "grouped CV by decorated sequence", RGBColor(238, 244, 235))
    add_metric(slide, Inches(3.82), Inches(3.6), Inches(2.6), Inches(1.55), "source Fail rows", "86", "rare event: 3.25% baseline", RGBColor(247, 235, 232))
    add_metric(slide, Inches(6.8), Inches(3.6), Inches(2.6), Inches(1.55), "engineered features", "161", "sequence, chemistry, position, runs", RGBColor(240, 243, 249))
    add_metric(slide, Inches(9.78), Inches(3.6), Inches(2.6), Inches(1.55), "best use", "ranking", "review prioritization, not hard calls", RGBColor(244, 239, 229))
    return slide


def slide_data_features(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title(slide, "Data and Feature Engineering", "Decorated sequence strings are converted into model-ready chemistry signals.")
    add_card(slide, Inches(0.75), Inches(1.58), Inches(3.55), Inches(1.1), "Primary endpoint", "Source Pass/Fail from cleaned sgRNA records; Fail is rare and noisy.", RGBColor(247, 239, 232))
    add_card(slide, Inches(4.75), Inches(1.58), Inches(3.55), Inches(1.1), "Auxiliary endpoints", "MainPeak purity regression and formula-derived Need_Review classification.", RGBColor(234, 244, 242))
    add_card(slide, Inches(8.75), Inches(1.58), Inches(3.55), Inches(1.1), "Parser detail", "+A/+C/+G/+U/+T are normalized as LNA tokens lA/lC/lG/lU/lT.", RGBColor(240, 243, 249))
    add_bullets(
        slide,
        Inches(0.85),
        Inches(3.15),
        Inches(5.2),
        Inches(2.45),
        [
            "Feature families: sequence size, base composition, motif/run structure, modification burden, positional chemistry, and parser completeness.",
            "Top marginal Pass/Fail signals are LNA burden/run features, especially lT/lA count and fraction signals.",
            "All model comparisons use grouped cross-validation to reduce sequence-level leakage.",
        ],
        size=15,
    )
    add_image(slide, FEATURE_PLOT_DIR / "token_lT_count.png", Inches(6.55), Inches(3.05), Inches(5.7))
    return slide


def slide_modeling_results(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title(slide, "Modeling Results", "Correlation pruning helps rare-fail ranking; continuous purity is easier to predict.")
    headers = [("Endpoint", Inches(0.95)), ("Recommended model", Inches(3.55)), ("Key performance", Inches(6.95))]
    for label, left in headers:
        box = add_textbox(slide, left, Inches(1.42), Inches(2.7), Inches(0.3))
        paragraph = box.text_frame.paragraphs[0]
        paragraph.text = label
        set_run(paragraph.runs[0], size=11, color=GREEN, bold=True)
    add_result_row(slide, Inches(1.86), "Source Pass/Fail", "71-feature correlation-pruned HGB", "PR-AUC 0.1139; ROC AUC 0.6529; Precision@5% 0.1353", GREEN)
    add_result_row(slide, Inches(2.72), "Full baseline", "161-feature HGB", "PR-AUC 0.1087; ROC AUC 0.6304; Precision@5% 0.1278", BLUE)
    add_result_row(slide, Inches(3.58), "Need_Review", "71-feature correlation-pruned HGB", "PR-AUC 0.4082; ROC AUC 0.7719; F1 0.3467; Precision@5% 0.4737", ORANGE)
    add_result_row(slide, Inches(4.44), "MainPeak", "full-feature ExtraTrees", "RMSE 6.0574; R2 0.7005; Pearson r 0.8370", TEAL)
    add_result_row(slide, Inches(5.30), "Compact panel", "shared 9-feature panel", "Pass/Fail PR-AUC 0.1078; Need_Review PR-AUC 0.3749; MainPeak RMSE 6.2635", RED)
    add_card(slide, Inches(0.95), Inches(6.18), Inches(11.35), Inches(0.64), "Interpretation", "The models are strongest as ranked triage tools. Default 0.5 hard calls underuse the signal because Fail prevalence is only 3.25%.", RGBColor(244, 239, 229))
    return slide


def slide_feature_and_purity(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title(slide, "Feature Insights and Purity Signal", "A compact panel preserves much of the cross-endpoint signal.")
    add_image(slide, OUTPUT_DIR / "mainpeak_regression_observed_vs_predicted.png", Inches(0.78), Inches(1.62), Inches(5.65))
    add_bullets(
        slide,
        Inches(6.85),
        Inches(1.58),
        Inches(5.45),
        Inches(1.62),
        [
            "MainPeak keeps quantitative purity information that binary Pass/Fail labels discard.",
            "Top-40 ExtraTrees-selected features nearly match the full model: RMSE 6.0998 vs 6.0574.",
        ],
        size=15,
    )
    chip_specs = [
        ("rna_count_middle", 6.9, 3.45, 1.55, GREEN),
        ("modified_token_position_std", 8.6, 3.45, 2.15, BLUE),
        ("token_rC_count", 10.9, 3.45, 1.42, ORANGE),
        ("lna_span_fraction", 6.9, 3.95, 1.55, TEAL),
        ("base_U_count", 8.6, 3.95, 1.28, RED),
        ("star_count_3p", 10.05, 3.95, 1.35, GREEN),
        ("token_lG_fraction", 6.9, 4.45, 1.62, BLUE),
        ("longest_rU_run", 8.7, 4.45, 1.48, ORANGE),
        ("longest_lC_run", 10.35, 4.45, 1.5, TEAL),
    ]
    for text, left, top, width, color in chip_specs:
        add_feature_chip(slide, text, Inches(left), Inches(top), Inches(width), color)
    add_card(slide, Inches(6.85), Inches(5.2), Inches(5.45), Inches(1.08), "Compact interpretation", "The 9-feature panel is the best lightweight cross-endpoint explanation set, while the 71-feature HGB remains the best Pass/Fail ranking option.", RGBColor(240, 243, 249))
    return slide


def slide_recommendations(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_title(slide, "Operational Recommendations", "Use the current models to prioritize review and guide data collection.")
    add_card(slide, Inches(0.82), Inches(1.55), Inches(3.55), Inches(1.25), "Risk ranking", "Deploy the 71-feature HGB score as a review-prioritization queue, not an automatic release decision.", RGBColor(234, 244, 242))
    add_card(slide, Inches(4.88), Inches(1.55), Inches(3.55), Inches(1.25), "Threshold setting", "Choose cutoffs from review budget or target precision, then calibrate on prospective batches.", RGBColor(244, 239, 229))
    add_card(slide, Inches(8.94), Inches(1.55), Inches(3.55), Inches(1.25), "Data growth", "Collect more true-Fail and low-MainPeak examples; these are the limiting labels.", RGBColor(247, 235, 232))
    add_image(slide, OUTPUT_DIR / "learning_curve.png", Inches(0.95), Inches(3.32), Inches(5.55))
    add_bullets(
        slide,
        Inches(7.0),
        Inches(3.35),
        Inches(5.15),
        Inches(2.5),
        [
            "Best source Pass/Fail model: 71-feature correlation-pruned HGB for top-risk enrichment.",
            "Best compact reporting view: revised 9-feature panel across Pass/Fail, Need_Review, and MainPeak.",
            "Best purity predictor: full-feature ExtraTrees, or top-40 ExtraTrees features when compactness matters.",
        ],
        size=15,
    )
    return slide


def build_deck() -> None:
    require_inputs()
    pd.read_csv(OUTPUT_DIR / "correlation_reduced_model_comparison.csv")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slides = [
        slide_problem(prs),
        slide_data_features(prs),
        slide_modeling_results(prs),
        slide_feature_and_purity(prs),
        slide_recommendations(prs),
    ]
    for number, slide in enumerate(slides, start=1):
        if number > 1:
            add_footer(slide, number)
    prs.save(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(slides)} slides")


if __name__ == "__main__":
    build_deck()