import csv

src = "data/raw/student_data.csv"
dst = "data/raw/student_data_fixed.csv"

with open(src, encoding="utf-8", errors="ignore", newline="") as f:
    rows = list(csv.reader(f))      # undoes the outer quoting

header = rows[0][0].split(";") if len(rows[0]) == 1 else rows[0]
header = [h.strip().strip('"') for h in header]

fixed = []
for r in rows[1:]:
    if not r:
        continue
    line = r[0] if len(r) == 1 else ";".join(r)
    fields = next(csv.reader([line], delimiter=";", quotechar='"'))
    fields = [x.strip().strip('"') for x in fields]
    if len(fields) == len(header):
        fixed.append(fields)

with open(dst, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(header)
    w.writerows(fixed)

print("Rows recovered:", len(fixed), "| Columns:", len(header))