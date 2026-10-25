@{
    # Run the complete default rule set, including informational diagnostics.
    # Test-Standards.ps1 treats every unsuppressed diagnostic as a failure.
    IncludeDefaultRules = $true
    Severity = @('Error', 'Warning', 'Information', 'ParseError')
    # The project requires PowerShell 7.6, whose native text encoding is UTF-8
    # without a BOM. This Windows PowerShell 5.1 compatibility rule is inapplicable.
    ExcludeRules = @('PSUseBOMForUnicodeEncodedFile')
    Rules = @{
        PSAvoidExclaimOperator = @{ Enable = $true }
        PSAvoidUsingDoubleQuotesForConstantString = @{ Enable = $true }
        PSUseCompatibleSyntax = @{ Enable = $true; TargetVersions = @('7.6') }
        PSUseConsistentParameterSetName = @{ Enable = $true }
        PSUseConsistentParametersKind = @{ Enable = $true }
        PSUseSingleValueFromPipelineParameter = @{ Enable = $true }
        # FullLanguage is required for the reviewed .NET/native interop helpers.
        # Legacy compatibility profiles stop at PS7.0/.NET Core 3.1 and cannot
        # validate the required PS7.6/.NET10 APIs. The parser and #requires checks
        # in Test-Standards enforce the actual target instead.
    }
}
