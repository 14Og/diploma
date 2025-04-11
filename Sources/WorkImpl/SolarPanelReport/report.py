import sys, os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.writers import FilesWriter
from nbconvert import PDFExporter, TemplateExporter

#  pdf conversion options

pdf = PDFExporter()
pdf.preprocessors = ["pre_pymarkdown.PyMarkdownPreprocessor"]
pdf.exclude_input = True

# PDFExporter.extra_template_basedirs = [
#     os.path.join(
#         os.path.dirname(os.path.realpath(__file__)), "templates", "hide_header"
#     )
# ]

path = os.path.dirname(__file__)


NOTEBOOK_ARGS_PATH = os.path.join(path, ".config_ipynb")
NOTEBOOK_PATH = os.path.join(path, "report.ipynb")


def create_report(panelpath: str, reportname: str):

    with open(NOTEBOOK_ARGS_PATH, "w") as f:
        f.write(" ".join([panelpath, reportname]))

    with open(NOTEBOOK_PATH) as nbfile:
        nb = nbformat.read(nbfile, as_version=nbformat.NO_CONVERT)

    print(
        f"Starting pdf export into: {panelpath} to {reportname}.pdf from {NOTEBOOK_PATH}"
    )

    ep = ExecutePreprocessor(timeout=100, kernel_name="python3")
    ep.preprocess(
        nb,
        {
            "metadata": {
                "title": "Report title",
                "path": path,
            },
        },
    )

    body, resources = pdf.from_notebook_node(nb)
    writer = FilesWriter()
    writer.build_directory = panelpath
    writer.write(body, resources, notebook_name=reportname)

    # runstr = f"jupyter nbconvert --execute {NOTEBOOK_PATH} --to pdf --no-input \
    # --template /home/nikolai/projects/pypon/SunSensorCalibration/SolarPanelReport/templates/hide_header \
    # --output {os.path.join(panelpath, 'report', reportname)}.pdf"

    # os.system(f"jupyter execute {NOTEBOOK_PATH}")
    # os.system(runstr)
