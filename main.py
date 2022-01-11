import cv2
import numpy as np
import pytesseract
import os


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def find_between_r(s, first, last):
    try:
        start = s.rindex(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""


template_id = 2

per = 100
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# roi = [[(354, 489), (637, 523), 'text', 'ref-name'],  #template_1
#          [(352, 527), (734, 554), 'text', 'ref-place'],
#          [(354, 564), (759, 634), 'text', 'ref-address'],
#     [(247, 762), (1532, 2267), 'text', 'free-text']]

roi = [[(174, 444), (539, 477), 'text', 'ref-name'], #template_2
       [(172, 479), (562, 512), 'text', 'ref-address'],
       [(170, 551), (440, 585), 'text', 'ref-date'],
       [(1102, 417), (1609, 604), 'text', 'free-text-pat'],
       [(147, 662), (1612, 1314), 'text', 'free-text-pat']]

# roi = [[(107, 372), (1632, 2292), 'text', 'free-text']] #template_3


# roi = [[(42, 427), (1619, 2269), 'text', 'free-text']]  # template_4

# folder location C:\Users\shaki\Desktop\template

imgQ = cv2.imread("blank-templates/blank-template-2.jpg")

h, w, c = imgQ.shape
# imgQ = cv2.resize(imgQ, (w // 3, h // 3))

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ, None)
# impKp1 = cv2.drawKeypoints(imgQ,kp1,None)

img = cv2.imread("template/test-template-2.jpg")

kp2, des2 = orb.detectAndCompute(img, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = list(bf.match(des2, des1))
matches.sort(key=lambda x: x.distance)
good = matches[:int(len(matches) * (per / 100))]
# imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:20], None, flags=2)

# cv2.imshow('1', imgMatch)

srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
imgScan = cv2.warpPerspective(img, M, (w, h))

imgShow = imgScan.copy()
imgMask = np.zeros_like(imgShow)

myData = []

for x, r in enumerate(roi):
    cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
    imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

    imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
    # to show cropped images
    # cv2.imshow(str(x), imgCrop)

    if r[2] == 'text':
        myData.append(pytesseract.image_to_string(imgCrop))
        # print(r[3] + ' ' + pytesseract.image_to_string(imgCrop))

    # cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)

# to show highlighted image
# imgShow = cv2.resize(imgShow, (w // 3, h // 3))
# cv2.imshow('1', imgShow)


if template_id == 1:
    ref_name = myData[0]
    ref_place = myData[1]
    ref_address = myData[2]

    free_text = myData[3]
    ref_date = find_between(free_text, "on", "for")
    ref_type = find_between(free_text, ref_date + "for", "and this")
    UR = find_between(free_text, "UR", "Name")
    Name = find_between(free_text, "Name", "DOB")
    DOB = find_between(free_text, "DOB", "Address")
    Address = find_between(free_text, "Address", "An")
    MRN = find_between(free_text, "MRN:", "Page")

    ref_date = ref_date.rstrip("\n")
    ref_type = ref_type.rstrip("\n")
    UR = UR.rstrip("\n")
    Name = Name.rstrip("\n")
    DOB = DOB.rstrip("\n")
    Address = Address.rstrip("\n")
    MRN = MRN.rstrip("\n")
    ref_name = ref_name.rstrip("\n")
    ref_place = ref_place.rstrip("\n")
    ref_address = ref_address.rstrip("\n")

    ref_date = ref_date.replace(":", "")
    ref_type = ref_type.replace(":", "")
    UR = UR.replace(":", "")
    Name = Name.replace(":", "")
    DOB = DOB.replace(":", "")
    Address = Address.replace(":", "")
    MRN = MRN.replace(":", "")
    ref_name = ref_name.replace(":", "")
    ref_place = ref_place.replace(":", "")
    ref_address = ref_address.replace(":", "")

    ref_date = ref_date.strip()
    ref_type = ref_type.strip()
    UR = UR.strip()
    Name = Name.strip()
    DOB = DOB.strip()
    Address = Address.strip()
    MRN = MRN.strip()
    ref_name = ref_name.strip()
    ref_place = ref_place.strip()
    ref_address = ref_address.strip()

    print("ref_name: " + ref_name)
    print("ref_place: " + ref_place)
    print("ref_address: " + ref_address)
    print("ref_date: " + ref_date)
    print("ref_type: " + ref_type)
    print("UR: " + UR)
    print("Name: " + Name)
    print("DOB: " + DOB)
    print("Address: " + Address)
    print("MRN: " + MRN)

if template_id == 2:
    ref_name = myData[0]
    ref_address = myData[1]
    ref_date = myData[2]

    free_text_patient_id = myData[3]
    free_text_patient_info = myData[4]

    patient_id = find_between(free_text_patient_id, "Patient ID:", "Accession")
    accession_number = find_between(free_text_patient_id, "Number:", "Reported")
    reported_on = find_between(free_text_patient_id, "Reported:", "\n")

    patient_name = find_between(free_text_patient_info, "Re:", "-")
    DOB = find_between(free_text_patient_info, "DOB:", "\n")
    address = find_between(free_text_patient_info, DOB + "\n", "\n")

    ref_name = ref_name.rstrip("\n")
    ref_address = ref_address.rstrip("\n")
    ref_date = ref_date.rstrip("\n")
    patient_id = patient_id.rstrip("\n")
    accession_number = accession_number.rstrip("\n")
    reported_on = reported_on.rstrip("\n")
    patient_name = patient_name.rstrip("\n")
    DOB = DOB.rstrip("\n")
    address = address.rstrip("\n")

    ref_name = ref_name.replace(":", "")
    ref_address = ref_address.replace(":", "")
    ref_date = ref_date.replace(":", "")
    patient_id = patient_id.replace(":", "")
    accession_number = accession_number.replace(":", "")
    reported_on = reported_on.replace(":", "")
    patient_name = patient_name.replace(":", "")
    DOB = DOB.replace(":", "")
    address = address.replace(":", "")

    ref_name = ref_name.strip()
    ref_address = ref_address.strip()
    ref_date = ref_date.strip()
    patient_id = patient_id.strip()
    accession_number = accession_number.strip()
    patient_name = patient_name.strip()
    DOB = DOB.strip()
    reported_on = reported_on.strip()
    address = address.strip()

    print("ref_name: " + ref_name)
    print("ref_address: " + ref_address)
    print("ref_date: " + ref_date)
    print("patient_id: " + patient_id)
    print("accession_number: " + accession_number)
    print("reported_on: " + reported_on)
    print("patient_name: " + patient_name)
    print("DOB: " + DOB)
    print("address: " + address)

if template_id == 3:
    free_text = myData[0]

    patient_name = find_between(free_text, "Patient:", "Date")
    patient_id = find_between(free_text, "Patient ID:", "DOB")
    DOB = find_between(free_text, "DOB:", "Gender")
    gender = find_between(free_text, "Gender:", "Reported by:")
    reported_by = find_between(free_text, "Reported by:", "Referred by")
    referred_by = find_between(free_text, "Referred by:", "Staff")
    scientific_officer = find_between(free_text, "Scientific Officer", "Reporting")
    reporting_mo = find_between(free_text, "Reporting MO", "\n")
    summary = find_between(free_text, "Summary", "Patient Events")
    patient_events = find_between(free_text, "Patient Events", "Conclusions")
    conclusions = find_between(free_text, "Conclusions", "Distribution")
    distribution = find_between(free_text, "Distribution", "Attachments")
    pat = find_between(free_text, "Pat:", "/")
    cr = find_between(free_text, "CR#", "Reported")

    patient_name = patient_name.rstrip("\n")
    patient_id = patient_id.rstrip("\n")
    DOB = DOB.rstrip("\n")
    gender = gender.rstrip("\n")
    reported_by = reported_by.rstrip("\n")
    referred_by = referred_by.rstrip("\n")
    scientific_officer = scientific_officer.rstrip("\n")
    reporting_mo = reporting_mo.rstrip("\n")
    summary = summary.rstrip("\n")
    patient_events = patient_events.rstrip("\n")
    conclusions = conclusions.rstrip("\n")
    distribution = distribution.rstrip("\n")
    pat = pat.rstrip("\n")
    cr = cr.rstrip("\n")

    patient_name = patient_name.replace(":", "")
    patient_id = patient_id.replace(":", "")
    DOB = DOB.replace(":", "")
    gender = gender.replace(":", "")
    reported_by = reported_by.replace(":", "")
    referred_by = referred_by.replace(":", "")
    scientific_officer = scientific_officer.replace(":", "")
    reporting_mo = reporting_mo.replace(":", "")
    summary = summary.replace(":", "")
    patient_events = patient_events.replace(":", "")
    conclusions = conclusions.replace(":", "")
    distribution = distribution.replace(":", "")
    pat = pat.replace(":", "")
    cr = cr.replace(":", "")

    patient_name = patient_name.strip()
    patient_id = patient_id.strip()
    DOB = DOB.strip()
    gender = gender.strip()
    reported_by = reported_by.strip()
    referred_by = referred_by.strip()
    scientific_officer = scientific_officer.strip()
    reporting_mo = reporting_mo.strip()
    summary = summary.strip()
    patient_events = patient_events.strip()
    conclusions = conclusions.strip()
    distribution = distribution.strip()
    pat = pat.strip()
    cr = cr.strip()

    print("patient_name: " + patient_name)
    print("patient_id: " + patient_id)
    print("DOB: " + DOB)
    print("gender: " + gender)
    print("reported_by: " + reported_by)
    print("referred_by: " + referred_by)
    print("scientific_officer: " + scientific_officer)
    print("reporting_mo: " + reporting_mo)

    summary = summary.split("\n")
    patient_events = patient_events.split("\n")

    for s in summary:
        if s == "" or s == " ":
            summary.remove(s)

    for p in patient_events:
        if p == "" or p == " ":
            patient_events.remove(p)

    for i,s in enumerate(summary):
        print("Summary "+str(i)+": "+s)

    for i,p in enumerate(patient_events):
        print("Patient Events "+str(i)+": "+p)

    print("conclusions: " + conclusions)
    print("distribution: " + distribution)
    print("pat: " + pat)
    print("cr: " + cr)

if template_id == 4:
    free_text = myData[0]

    referrer = find_between(free_text, "Referrer:", "-")
    referrer_id = find_between(free_text, "- ", "\n")
    ref_phone = find_between(free_text, "Phone:", "Address")
    ref_address = find_between(free_text, "Address:", "Referral date")
    patient_name = find_between(free_text, "Patient name:", "Patient DOB")
    DOB = find_between(free_text, "Patient DOB:", "Address:")
    address = find_between(free_text, str(DOB)+"Address:", "Phone")
    phone = find_between(free_text, str(address)+"Phone:", "\n")
    medical_history = find_between(free_text, "Medical History", "Medications")
    medication = find_between(free_text, "Medications", "Dr")

    referrer = referrer.rstrip("\n")
    referrer_id = referrer_id.rstrip("\n")
    ref_phone = ref_phone.rstrip("\n")
    ref_address = ref_address.rstrip("\n")
    patient_name = patient_name.rstrip("\n")
    DOB = DOB.rstrip("\n")
    address = address.rstrip("\n")
    phone = phone.rstrip("\n")
    medical_history = medical_history.rstrip("\n")
    medication = medication.rstrip("\n")

    referrer = referrer.replace(":", "")
    referrer_id = referrer_id.replace(":", "")
    ref_phone = ref_phone.replace(":", "")
    ref_address = ref_address.replace(":", "")
    patient_name = patient_name.replace(":", "")
    DOB = DOB.replace(":", "")
    address = address.replace(":", "")
    phone = phone.replace(":", "")
    medical_history = medical_history.replace(":", "")
    medication = medication.replace(":", "")

    referrer = referrer.strip()
    referrer_id = referrer_id.strip()
    ref_phone = ref_phone.strip()
    ref_address = ref_address.strip()
    patient_name = patient_name.strip()
    DOB = DOB.strip()
    phone = phone.strip()
    address = address.strip()
    medical_history = medical_history.strip()
    medication = medication.strip()


    print("referrer: "+referrer)
    print("referrer_id: " + referrer_id)
    print("ref_phone: " + ref_phone)
    print("ref_address: " + ref_address)
    print("patient_name: " + patient_name)
    print("DOB: " + DOB)
    print("phone: " + phone)
    print("address: " + address)

    medical_history = medical_history.split("-")
    medication = medication.split("-")
    print(medical_history)
    print(medication)


# cv2.imshow('Output', imgScan)
cv2.waitKey(0)
