import re
import pandas as pd
import numpy as np
from datetime import timedelta, date, datetime
import calendar

# Variations of dates that the parser can capture
year_variations = ['year', 'years', 'yrs']
day_variations = ['days', 'day']
minute_variations = ['minute', 'minutes', 'mins']
hour_variations = ['hrs', 'hours', 'hour']
week_variations = ['weeks', 'week', 'wks']
month_variations = ['month', 'months']

# Variables used for RegEx Matching
day_names = 'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
month_names_long = 'january|february|march|april|may|june|july|august|september|october|november|december'
month_names = month_names_long + '|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec'
day_nearest_names = 'today|yesterday|tomorrow|tonight|tonite'
numbers = "(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
                    eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
                    eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
                    ninety|hundred|thousand)"
re_dmy = '(' + "|".join(day_variations + minute_variations + year_variations + week_variations + month_variations) + ')'
re_duration = '(before|after|earlier|later|ago|from\snow)'
re_year = "(19|20)\d{2}|^(19|20)\d{2}"
re_timeframe = 'this|coming|next|following|previous|last|end\sof\sthe'
re_ordinal = 'st|nd|rd|th|first|second|third|fourth|fourth|' + re_timeframe
re_time = '(?P<hour>\d{1,2})(\:(?P<minute>\d{1,2})|(?P<convention>am|pm))'
re_separator = 'of|at|on'

# A list tuple of regular expressions / parser fn to match
# The order of the match in this list matters, So always start with the widest match and narrow it down
regex = [
    (re.compile(
        r'''
        (
            ((?P<dow>%s)[,\s]\s*)? #Matches Monday, 12 Jan 2012, 12 Jan 2012 etc
            (?P<day>\d{1,2}) # Matches a digit
            (%s)?
            [,\s]*[-\s] # One or more space
            (?P<month>%s) # Matches any month name
            [,\s]*[-\s] # Space
            (?P<year>%s) # Year
            ((\s|,\s|\s(%s))?\s*(%s))?
        )
        ''' % (day_names, re_ordinal, month_names, re_year, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year":
                                int(m.group('year')) if m.group('year') else None,
                            "month": hashmonths[m.group('month').strip().lower()],
                            "day": int(m.group('day')) if m.group('day') else None},
                           convertTimetoHourMinute(
                               m.group('hour'),
                               m.group('minute'),
                               m.group('convention')
                           ))
    ),
    (re.compile(
        r'''
        (
            ((?P<dow>%s)[,\s][-\s]*)? #Matches Monday, Jan 12 2012, Jan 12 2012 etc
            (?P<month>%s) # Matches any month name
            [,\s]*[-\s] # Space
            ((?P<day>\d{1,2})) # Matches a digit
            (%s)?
            ([,\s]*[-\s](?P<year>%s))? # Year
            ((\s|,\s|\s(%s))?\s*(%s))?
        )
        ''' % (day_names, month_names, re_ordinal, re_year, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year":
                                int(m.group('year')) if m.group('year') else None,
                            "month": hashmonths[m.group('month').strip().lower()],
                            "day": int(m.group('day')) if m.group('day') else None}
                           , convertTimetoHourMinute(
         m.group('hour'),
         m.group('minute'),
         m.group('convention')
     ))
    ),
    (re.compile(
        r'''
        (
            (?P<month>%s) # Matches any month name
            [,\s]*[-\s] # One or more space
            (?P<day>\d{1,2}) # Matches a digit
            (%s)?
            [,\s]*[-\s]\s*?
            (?P<year>%s) # Year
            ((\s|,\s|\s(%s))?\s*(%s))?
        )
        ''' % (month_names, re_ordinal, re_year, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year":
                                int(m.group('year')) if m.group('year') else None,
                            "month": hashmonths[m.group('month').strip().lower()],
                            "day": int(m.group('day')) if m.group('day') else None},
                           convertTimetoHourMinute(
                               m.group('hour'),
                               m.group('minute'),
                               m.group('convention')
                           ))
    ),
    (re.compile(
        r'''
        (
            ((?P<number>\d+|(%s[-\s]?)+)\s)? # Matches any number or string 25 or twenty five
            (?P<unit>%s)s?\s # Matches days, months, years, weeks, minutes
            (?P<duration>%s) # before, after, earlier, later, ago, from now
            (\s*(?P<base_time>(%s)))?
            ((\s|,\s|\s(%s))?\s*(%s))?
        )
        ''' % (numbers, re_dmy, re_duration, day_nearest_names, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: (dateFromDuration(
         m.group('number'),
         m.group('unit').lower(),
         m.group('duration').lower(),
         m.group('base_time')
     ), convertTimetoHourMinute(
         m.group('hour'),
         m.group('minute'),
         m.group('convention')
     ))
    ),
    (re.compile(
        r'''
        (
            (?P<ordinal>%s) # First quarter of 2014
            \s+
            quarter\sof
            \s+
            (?P<year>%s)
        )
        ''' % (re_ordinal, re_year),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: dateFromQuarter(base_date,
                                          hashordinals[m.group('ordinal').lower()],
                                          int(m.group('year')) if m.group('year') else None
                                          )
    ),
    (re.compile(
        r'''
        (
            (?P<ordinal_value>\d+)
            (?P<ordinal>%s) # 1st January 2012
            ((\s|,\s|\s(%s))?\s*)?
            (?P<month>%s)
            ([,\s]\s*(?P<year>%s))?
        )
        ''' % (re_ordinal, re_separator, month_names, re_year),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year":
                                int(m.group('year')) if m.group('year') else None,
                            "month": int(hashmonths[m.group('month').lower()]) if m.group('month') else None,
                            "day": int(m.group('ordinal_value')) if m.group('ordinal_value') else None},
                           {"hours": None, "minutes": None}
                           )
    ),
    (re.compile(
        r'''
        (
            (?P<month>%s)
            \s+
            (?P<ordinal_value>\d+)
            (?P<ordinal>%s) # January 1st 2012
            ([,\s]\s*(?P<year>%s))?
        )
        ''' % (month_names, re_ordinal, re_year),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year":
                                int(m.group('year')) if m.group('year') else None,
                            "month": int(hashmonths[m.group('month').lower()]) if m.group('month') else None,
                            "day": int(m.group('ordinal_value')) if m.group('ordinal_value') else None},
                           {"hours": None, "minutes": None}
                           )
    ),
    (re.compile(
        r'''
        ((?P<time>%s))? # this, next, following, previous, last
        [-\s]*
        ((?P<number>\d+|(%s[-\s]?)+))?
        \s+
        (?P<dmy>%s) # year, day, week, month, night, minute, min
        ((\s|,\s|\s(%s))?\s*(%s))?
        ''' % (re_timeframe, numbers, re_dmy, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE),
    ),
     lambda m, base_date: (dateFromRelativeWeekYear(
         base_date,
         m.group('time'),
         m.group('dmy'),
         m.group('number')
     ), convertTimetoHourMinute(
         m.group('hour'),
         m.group('minute'),
         m.group('convention')
     ))
    ),
    (re.compile(
        r'''
        (?P<time>%s) # this, next, following, previous, last
        \s+
        (?P<dow>%s) # mon - fri
        ((\s|,\s|\s(%s))?\s*(%s))?
        ''' % (re_timeframe, day_names, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE),
    ),
     lambda m, base_date: (dateFromRelativeDay(
         base_date,
         m.group('time'),
         m.group('dow')
     ), convertTimetoHourMinute(
         m.group('hour'),
         m.group('minute'),
         m.group('convention')
     ))
    ),
    (re.compile(
        r'''
        (
            (?P<day>\d{1,2}) # Day, Month
            (%s)?
            [,\s]*[-\s] # One or more space
            (?P<month>%s)
        )
        ''' % (re_ordinal, month_names),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year": None,
                            "month": hashmonths[m.group('month').strip().lower()],
                            "day": int(m.group('day')) if m.group('day') else None}, {"hours": None, "minutes": None}
                           )
    ),
    (re.compile(
        r'''
        (
            (?P<month>%s) # Month, year
            [,\s]*[-\s] # One or more space
            ((?P<year>\d{2,4})\b) # Matches a digit January 12
        )
        ''' % (month_names),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({
                               "year": int(m.group('year')),
                               "month": hashmonths[m.group('month').strip().lower()],
                               "day": None}
     , {"hours": None, "minutes": None})
    ),
    (re.compile(
        r'''
        (
            (?P<month>%s) # Month, day
            [-\s] # One or more space
            ((?P<day>\d{1,2})\b) # Matches a digit January 12
            (%s)?
        )
        ''' % (month_names, re_ordinal),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year": None,
                            "month": hashmonths[m.group('month').strip().lower()],
                            "day": int(m.group('day')) if m.group('day') else None}, {"hours": None, "minutes": None}
                           )
    ),
    (re.compile(
        r'''
        (
            (?P<month>\d{1,2}) # MM/DD or MM/DD/YYYY
            /
            ((?P<day>\d{1,2}))
            (/(?P<year>%s))?
        )
        ''' % (re_year),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year":
                                int(m.group('year')) if m.group('year') else None,
                            "month": int(m.group('month').strip()),
                            "day": int(m.group('day'))}
                           , {"hours": None, "minutes": None})
    ),
    (re.compile(
        r'''
        (?P<adverb>%s) # today, yesterday, tomorrow, tonight
        ((\s|,\s|\s(%s))?\s*(%s))?
        ''' % (day_nearest_names, re_separator, re_time),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: (dateFromAdverb(
         base_date,
         m.group('adverb')
     ), convertTimetoHourMinute(
         m.group('hour'),
         m.group('minute'),
         m.group('convention')
     ))
    ),
    (re.compile(
        r'''
        (?P<named_day>%s) # Mon - Sun
        ''' % (day_names),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: (this_week_day(
         base_date,
         hashweekdays[m.group('named_day').lower()]
     ), {"hours": None, "minutes": None})
    ),
    (re.compile(
        r'''
        (?P<year>%s) # Year
        ''' % (re_year),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({"year": int(m.group('year')), "month": None, "day": None}, {"hours": None, "minutes": None})
    ),
    (re.compile(
        r'''
        (?P<month>%s) # Month
        ''' % (month_names_long),
        (re.VERBOSE | re.IGNORECASE)
    ),
     lambda m, base_date: ({
                               "year": None,
                               "month": hashmonths[m.group('month').lower()],
                               "day": None}
     , {"hours": None, "minutes": None})
    ),
    (re.compile(
        r'''
        (%s) # Matches time 12:00
        ''' % (re_time),
        (re.VERBOSE | re.IGNORECASE),
    ),
     lambda m, base_date: ({"year": None, "month": None, "day": None},
                           convertTimetoHourMinute(
                               m.group('hour'),
                               m.group('minute'),
                               m.group('convention')
                           ))
    ),
    (re.compile(
        r'''
        (
            (?P<hour>\d+) # Matches 12 hours, 2 hrs
            \s+
            (%s)
        )
        ''' % ('|'.join(hour_variations)),
        (re.VERBOSE | re.IGNORECASE),
    ),
     lambda m, base_date: ({"year": None, "month": None, "day": None}, {"hours": int(m.group('hour')), "minutes": None})

    )
]


# Hash of numbers
# Append more number to modify your match
def hashnum(number):
    if re.match(r'one|^a\b', number, re.IGNORECASE):
        return 1
    if re.match(r'two', number, re.IGNORECASE):
        return 2
    if re.match(r'three', number, re.IGNORECASE):
        return 3
    if re.match(r'four', number, re.IGNORECASE):
        return 4
    if re.match(r'five', number, re.IGNORECASE):
        return 5
    if re.match(r'six', number, re.IGNORECASE):
        return 6
    if re.match(r'seven', number, re.IGNORECASE):
        return 7
    if re.match(r'eight', number, re.IGNORECASE):
        return 8
    if re.match(r'nine', number, re.IGNORECASE):
        return 9
    if re.match(r'ten', number, re.IGNORECASE):
        return 10
    if re.match(r'eleven', number, re.IGNORECASE):
        return 11
    if re.match(r'twelve', number, re.IGNORECASE):
        return 12
    if re.match(r'thirteen', number, re.IGNORECASE):
        return 13
    if re.match(r'fourteen', number, re.IGNORECASE):
        return 14
    if re.match(r'fifteen', number, re.IGNORECASE):
        return 15
    if re.match(r'sixteen', number, re.IGNORECASE):
        return 16
    if re.match(r'seventeen', number, re.IGNORECASE):
        return 17
    if re.match(r'eighteen', number, re.IGNORECASE):
        return 18
    if re.match(r'nineteen', number, re.IGNORECASE):
        return 19
    if re.match(r'twenty', number, re.IGNORECASE):
        return 20
    if re.match(r'thirty', number, re.IGNORECASE):
        return 30
    if re.match(r'forty', number, re.IGNORECASE):
        return 40
    if re.match(r'fifty', number, re.IGNORECASE):
        return 50
    if re.match(r'sixty', number, re.IGNORECASE):
        return 60
    if re.match(r'seventy', number, re.IGNORECASE):
        return 70
    if re.match(r'eighty', number, re.IGNORECASE):
        return 80
    if re.match(r'ninety', number, re.IGNORECASE):
        return 90
    if re.match(r'hundred', number, re.IGNORECASE):
        return 100
    if re.match(r'thousand', number, re.IGNORECASE):
        return 1000


# Convert strings to numbers
def convert_string_to_number(value):
    if value == None:
        return 1
    if isinstance(value, int):
        return value
    if value.isdigit():
        return int(value)
    num_list = map(lambda s: hashnum(s), re.findall(numbers + '+', value, re.IGNORECASE))
    return sum(num_list)


# Convert time to hour, minute
def convertTimetoHourMinute(hour, minute, convention):
    if hour is None:
        hour = 0
    if minute is None:
        minute = 0
    if convention is None:
        convention = 'am'

    hour = int(hour)
    minute = int(minute)

    if convention == 'pm':
        hour += 12

    return {'hours': hour, 'minutes': minute}


# Quarter of a year
def dateFromQuarter(base_date, ordinal, year):
    interval = 3
    month_start = interval * (ordinal - 1)
    if month_start < 0:
        month_start = 9
    month_end = month_start + interval
    if month_start == 0:
        month_start = 1
    if (year is None):
        year = base_date.year
    return ({"year": year, "month": month_start, "day": 1}, {'hours': 0, 'minutes': 0})


# Converts relative day to time
# this tuesday, last tuesday
def dateFromRelativeDay(base_date, time, dow):
    # Reset date to start of the day
    base_date = datetime(base_date.year, base_date.month, base_date.day)
    time = time.lower()
    dow = dow.lower()
    if time == 'this' or time == 'coming':
        # Else day of week
        num = hashweekdays[dow]
        return this_week_day(base_date, num)
    elif time == 'last' or time == 'previous':
        # Else day of week
        num = hashweekdays[dow]
        return previous_week_day(base_date, num)
    elif time == 'next' or time == 'following':
        # Else day of week
        num = hashweekdays[dow]
        return next_week_day(base_date, num)


# Converts relative day to time
# this tuesday, last tuesday
def dateFromRelativeWeekYear(base_date, time, dow, ordinal_p=1):
    # If there is an ordinal (next 3 weeks) => return a start and end range
    # Reset date to start of the day
    ordinal = convert_string_to_number(ordinal_p)
    # print(time,dow,ordinal)
    if(dow is None):
        dow=""
    else:
        dow=dow.lower()
    d = datetime(base_date.year, base_date.month, base_date.day)
    if dow in year_variations:
        if time == 'this' or time == 'coming':
            return {"year": d.year, "month": 1, "day": 1}
        elif time == 'last' or time == 'previous' or (time is None and ordinal_p is not None):
            return {"year": d.year - ordinal, "month": d.month, "day": 1}
        elif time == 'next' or time == 'following':
            return {"year": d.year + ordinal, "month": d.month, "day": d.day}
        elif time == 'end of the':
            return {"year": d.year, "month": 12, "day": 31}
    elif dow in month_variations:
        if time == 'this':
            return {"year": d.year, "month": d.month, "day": d.day}
        elif time == 'last' or time == 'previous' or (time is None and ordinal_p is not None):
            mon = d.month - ordinal
            if (mon < 1):
                mon = 1
            return {"year": d.year, "month": mon, "day": d.day}
        elif time == 'next' or time == 'following':
            mon = d.month + ordinal
            if (mon > 12):
                mon = 12
            return {"year": d.year, "month": mon, "day": d.day}
        elif time == 'end of the':
            return {"year": d.year, "month": d.month, "day": calendar.monthrange(d.year, d.month)[1]}
    elif dow in week_variations:
        if time == 'this':
            temp = d - timedelta(days=d.weekday())
            return {"year": temp.year, "month": temp.month, "day": temp.day}
        elif time == 'last' or time == 'previous' or (time is None and ordinal_p is not None):
            temp = d - timedelta(weeks=ordinal)
            return {"year": temp.year, "month": temp.month, "day": temp.day}
        elif time == 'next' or time == 'following':
            temp = d + timedelta(weeks=ordinal)
            return {"year": temp.year, "month": temp.month, "day": temp.day}
        elif time == 'end of the':
            day_of_week = base_date.weekday()
            temp = d + timedelta(days=6 - d.weekday())
            return {"year": temp.year, "month": temp.month, "day": temp.day}
    elif dow in day_variations:
        if time == 'this':
            return {"year": d.year, "month": d.month, "day": d.day}
        elif time == 'last' or time == 'previous' or (time is None and ordinal_p is not None):
            temp = d - timedelta(days=ordinal)
            return {"year": temp.year, "month": temp.month, "day": temp.day}
        elif time == 'next' or time == 'following':
            temp = d + timedelta(days=ordinal)
            return {"year": temp.year, "month": temp.month, "day": temp.day}
        elif time == 'end of the':
            return {"year": d.year, "month": d.month, "day": d.day}
    else:
        return {"year": None, "month": None, "day": None}


# Convert Day adverbs to dates
# Tomorrow => Date
# Today => Date
def dateFromAdverb(base_date, name):
    # Reset date to start of the day
    d = datetime(base_date.year, base_date.month, base_date.day)
    if name.lower() == 'today' or name == 'tonite' or name == 'tonight':
        temp = d
        return {"year": temp.year, "month": temp.month, "day": temp.day}
    elif name == 'yesterday':
        temp = d - timedelta(days=1)
        return {"year": temp.year, "month": temp.month, "day": temp.day}
    elif name == 'tomorrow' or name == 'tom':
        temp = d + timedelta(days=1)
        return {"year": temp.year, "month": temp.month, "day": temp.day}
    else:
        return {"year": None, "month": None, "day": None}


# Find dates from duration
# Eg: 20 days from now
# Doesnt support 20 days from last monday
def dateFromDuration(base_date, numberAsString, unit, duration, base_time=None):
    # Check if query is `2 days before yesterday` or `day before yesterday`
    if base_time != None:
        base_date = dateFromAdverb(base_date, base_time)
    num = convert_string_to_number(numberAsString)
    if unit in day_variations:
        args = {'days': num}
    elif unit in minute_variations:
        args = {'minutes': num}
    elif unit in week_variations:
        args = {'weeks': num}
    elif unit in month_variations:
        args = {'days': 365 * num / 12}
    elif unit in year_variations:
        args = {'years': num}
    if duration == 'ago' or duration == 'before' or duration == 'earlier':
        if ('years' in args):
            return {"year": base_date.year - args['years'], "month": base_date.month, "day": base_date.day}
        temp = base_date - timedelta(**args)
        return {"year": temp.year, "month": temp.month, "day": temp.day}
    elif duration == 'after' or duration == 'later' or duration == 'from now':
        if ('years' in args):
            return {"year": base_date.year + args['years'], "month": base_date.month, "day": base_date.day}
        temp = base_date + timedelta(**args)
        return {"year": temp.year, "month": temp.month, "day": temp.day}
    else:
        return {"year": None, "month": None, "day": None}


# Finds coming weekday
def this_week_day(base_date, weekday):
    day_of_week = base_date.weekday()
    # If today is Tuesday and the query is `this monday`
    # We should output the next_week monday
    if day_of_week > weekday:
        return next_week_day(base_date, weekday)
    start_of_this_week = base_date - timedelta(days=day_of_week + 1)
    day = start_of_this_week + timedelta(days=1)
    while day.weekday() != weekday:
        day = day + timedelta(days=1)
    return {"year": day.year, "month": day.month, "day": day.day}


# Finds coming weekday
def previous_week_day(base_date, weekday):
    day = base_date - timedelta(days=1)
    while day.weekday() != weekday:
        day = day - timedelta(days=1)
    return {"year": day.year, "month": day.month, "day": day.day}


def next_week_day(base_date, weekday):
    day_of_week = base_date.weekday()
    end_of_this_week = base_date + timedelta(days=6 - day_of_week)
    day = end_of_this_week + timedelta(days=1)
    while day.weekday() != weekday:
        day = day + timedelta(days=1)
    return {"year": day.year, "month": day.month, "day": day.day}


# Mapping of Month name and Value
hashmonths = {
    'january': 1,
    'jan': 1,
    'february': 2,
    'feb': 2,
    'march': 3,
    'mar': 3,
    'april': 4,
    'apr': 4,
    'may': 5,
    'june': 6,
    'jun': 6,
    'july': 7,
    'jul': 7,
    'august': 8,
    'aug': 8,
    'september': 9,
    'sep': 9,
    'october': 10,
    'oct': 10,
    'november': 11,
    'nov': 11,
    'december': 12,
    'dec': 12
}

# Days to number mapping
hashweekdays = {
    'monday': 0,
    'mon': 0,
    'tuesday': 1,
    'tue': 1,
    'wednesday': 2,
    'wed': 2,
    'thursday': 3,
    'thu': 3,
    'friday': 4,
    'fri': 4,
    'saturday': 5,
    'sat': 5,
    'sunday': 6,
    'sun': 6
}

# Ordinal to number
hashordinals = {
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth': 4,
    'forth': 4,
    'last': -1
}


# Parses date
def datetime_parsing(text, base_date=datetime.now(),max_def_count=0):
    matches = []
    found_array = []

    # Find the position in the string
    # print(text)
    text = text.replace(".", "")
    span_se = np.array([])
    for r, fn in regex:
        for m in r.finditer(text):
            temp = (m.group(), fn(m, base_date))
            test_se = [0]
            if (span_se.shape[0] > 0):
                # print(span_se,m.start(),m.end())
                test_se = np.logical_and(span_se[:, 0] >= m.start(), span_se[:, 1] <= m.end())
            if (np.any(test_se)):
                idx = np.nonzero(test_se)[0][0]
                # print(np.nonzero(test_se),test_se)
                matches[idx] = temp
                span_se[idx, :] = np.array([m.start(), m.end()])
            else:
                matches.append(temp)
                if (span_se.shape[0] == 0):
                    span_se = np.array([m.start(), m.end()]).reshape((1, 2))
                else:
                    span_se = np.vstack((span_se, np.array([m.start(), m.end()])))
    #             if(text=="today"):
    #                 print(temp)

    # Wrap the matched text with TAG element to prevent nested selections
    date_dict = {"year": None, "month": None, "day": None}
    time_dict = {"hours": None, "minutes": None}
    spans = []
    res_dates = []
    # print(matches)
    for match, value in matches:
        date_dict = {"year": None, "month": None, "day": None}
        time_dict = {"hours": None, "minutes": None}
        # print(match,value,base_date)
        subn = re.subn('(?!<TAG[^>]*?>)' + match + '(?![^<]*?</TAG>)', '<TAG>' + match + '</TAG>', text)
        text = subn[0]
        isSubstituted = subn[1]
        if isSubstituted != 0:
            new_flag, d, t = update_datetime(date_dict, time_dict, value)
            # if (new_flag):
            date_dict = d
            time_dict = t
            temp_date = get_datetime(base_date, date_dict, time_dict,max_def_count=max_def_count)
            if (temp_date is not None):
                res_dates.append(temp_date)
                spans.append(match)
            # else:
            date_dict = d
            time_dict = t
            # date_dict = {"year": None, "month": None, "day": None}
            # time_dict = {"hours": None, "minutes": None}

            # found_array.appen((match,value))
    # temp_date = get_datetime(base_date, date_dict, time_dict,max_def_count=max_def_count)
    # if (temp_date is not None):
    #     res_dates.append(temp_date)
    #     spans.append(match)
    return res_dates,spans


def update_datetime(date_dict, time_dict, value):
    res_date = {"year": date_dict["year"], "month": date_dict["month"], "day": date_dict["day"]}
    res_time = {"hours": time_dict["hours"], "minutes": time_dict["minutes"]}
    d = value[0]
    t = value[1]
    # print("DATE",d,"TIME",t)
    new_flag = False
    if (d is not None):
        if (d["year"] is not None):
            if (date_dict["year"] is not None and date_dict["year"] != d["year"]):
                new_flag = True
            res_date["year"] = d["year"]
        if (d["month"] is not None):
            if (date_dict["month"] is not None and date_dict["month"] != d["month"]):
                new_flag = True
            res_date["month"] = d["month"]
        if (d["day"] is not None):
            if (date_dict["day"] is not None and date_dict["day"] != d["day"]):
                new_flag = True
            res_date["day"] = d["day"]
    if (t is not None):
        if (t["hours"] is not None):
            # if (time_dict["hours"] is None):
            res_time["hours"] = t["hours"]
        if (t["minutes"] is not None):
            # if (time_dict["minutes"] is None):
            res_time["minutes"] = t["minutes"]

    return new_flag, res_date, res_time


def get_datetime(base_date, date_dict, time_dict, max_def_count=3):
    # print(date_dict,time_dict)
    res_date = {"year": date_dict["year"], "month": date_dict["month"], "day": date_dict["day"]}
    res_time = {"hours": time_dict["hours"], "minutes": time_dict["minutes"]}
    def_count = 0

    if (date_dict["year"] is None or date_dict["year"] < 1900):
        res_date["year"] = base_date.year
        def_count += 1
    if (date_dict["month"] is None):
        if (def_count > 0):
            res_date["month"] = 1
        else:
            res_date["month"] = base_date.month
        def_count += 1
    else:
        if (date_dict["month"] > 12):
            res_date["month"] = 12
        elif (date_dict["month"] < 1):
            res_date["month"] = 1

    if (date_dict["day"] is None):
        res_date["day"] = 1
        def_count += 1
    else:
        tot_days = pd.Period(datetime(res_date["year"], res_date["month"], 1), 'M').days_in_month
        if (date_dict["day"] > tot_days):
            res_date["day"] = tot_days
        elif (date_dict["day"] < 1):
            res_date["day"] = 1

    if (def_count > max_def_count):  # If all date fields are None
        return None
    else:
        if (time_dict["hours"] is None):
            res_time["hours"] = 0
        if (time_dict["minutes"] is None):
            res_time["minutes"] = 0
        return (datetime(res_date["year"], res_date["month"], res_date["day"]) + timedelta(hours=res_time["hours"],
                                                                                              minutes=res_time[
                                                                                                  "minutes"]))