"""Open the report in Word: update the table of contents, save, and export a PDF preview."""
import sys
from pathlib import Path

import win32com.client

docx = Path(sys.argv[1]).resolve()
word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
try:
    d = word.Documents.Open(str(docx))
    for t in d.TablesOfContents:
        t.Update()
    d.Save()
    d.ExportAsFixedFormat(str(docx.with_suffix(".pdf")), 17)  # 17 = wdExportFormatPDF
    print("pages:", d.ComputeStatistics(2))
    d.Close(False)
finally:
    word.Quit()
