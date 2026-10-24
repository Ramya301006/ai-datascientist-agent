from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet


class ReportGenerator:

    @staticmethod
    def generate_report(
        analysis,
        accuracy,
        feature_importance
    ):

        filename = "reports/report.pdf"

        doc = SimpleDocTemplate(
            filename
        )

        styles = getSampleStyleSheet()

        content = []

        content.append(
            Paragraph(
                "AI Data Scientist Report",
                styles["Title"]
            )
        )

        content.append(
            Spacer(1, 20)
        )

        content.append(
            Paragraph(
                f"Rows: {analysis['rows']}",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Columns: {analysis['columns']}",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Accuracy: {accuracy:.2f}%",
                styles["Normal"]
            )
        )

        content.append(
            Spacer(1, 20)
        )

        content.append(
            Paragraph(
                "Top Features",
                styles["Heading2"]
            )
        )

        for key, value in feature_importance.items():

            content.append(
                Paragraph(
                    f"{key} : {value}",
                    styles["Normal"]
                )
            )

        doc.build(content)

        return filename