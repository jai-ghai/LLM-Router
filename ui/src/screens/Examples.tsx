import { CaretDown, CaretUp, Image as ImageIcon } from '@phosphor-icons/react'
import { Alert, Button, Collapse, IconButton, Stack, Table, TableBody, TableCell, TableHead, TableRow, TextField } from '@mui/material'
import { Fragment, useEffect, useState } from 'react'
import { api, describeError } from '../api'
import { DecisionCard } from '../components/DecisionCard'
import { ImagePicker } from '../components/ImagePicker'
import { MetadataFieldsEditor } from '../components/MetadataFieldsEditor'
import { Mono, PageHead, Panel, SectionHead, StateTag } from '../components/Primitives'
import { PALETTE } from '../theme'
import type { Decision, ExampleRow } from '../types'

/** Data > Examples: add an example (metadata + text + optional image), see what already exists,
 * and expand any row to try resolve() on it right here (the same DecisionCard the Resolve
 * screen uses). */
export function Examples() {
  const [rows, setRows] = useState<ExampleRow[]>([])
  const [metadata, setMetadata] = useState<Record<string, string>>({})
  const [text, setText] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [tryResults, setTryResults] = useState<Record<string, Decision | string>>({})

  async function load() {
    try {
      setRows((await api.examples()).examples)
    } catch (err) {
      setError(describeError(err))
    }
  }

  useEffect(() => {
    void load()
  }, [])

  async function addExample() {
    if (Object.keys(metadata).length === 0 && !text.trim()) {
      return setError('Add at least one metadata field or some text.')
    }
    setSaving(true)
    setError('')
    try {
      await api.addExample({ metadata, text, image })
      setMetadata({})
      setText('')
      setImage(null)
      await load()
    } catch (err) {
      setError(describeError(err))
    } finally {
      setSaving(false)
    }
  }

  async function tryResolve(row: ExampleRow) {
    try {
      const decision = await api.resolve({ metadata: row.metadata, text: row.text, image: null })
      setTryResults((prev) => ({ ...prev, [row.example_id]: decision }))
    } catch (err) {
      setTryResults((prev) => ({ ...prev, [row.example_id]: describeError(err) }))
    }
  }

  return (
    <>
      <PageHead title="Examples" lede="Add text, metadata, and optionally an image. Each one is checked against the rules immediately." />
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      <Stack spacing={2}>
        <Panel title="Add an example">
          <Stack spacing={1.5}>
            <SectionHead title="Metadata" />
            <MetadataFieldsEditor value={metadata} onChange={setMetadata} />
            <SectionHead title="Text" />
            <TextField value={text} onChange={(e) => setText(e.target.value)} multiline rows={2} fullWidth placeholder="What is this request about?" />
            <ImagePicker value={image} onChange={setImage} />
            <Button variant="contained" onClick={() => void addExample()} disabled={saving} sx={{ alignSelf: 'flex-start' }}>
              Add example
            </Button>
          </Stack>
        </Panel>

        <Panel title={`${rows.length} stored`} dense>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell />
                <TableCell>Metadata</TableCell>
                <TableCell>Text</TableCell>
                <TableCell>Covered by a rule?</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((row) => (
                <Fragment key={row.example_id}>
                  <TableRow>
                    <TableCell>
                      <IconButton
                        size="small"
                        aria-label={`${expanded === row.example_id ? 'Hide' : 'Show'} details for ${row.example_id}`}
                        onClick={() => setExpanded(expanded === row.example_id ? null : row.example_id)}
                      >
                        {expanded === row.example_id ? <CaretUp size={16} /> : <CaretDown size={16} />}
                      </IconButton>
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
                        {Object.entries(row.metadata).map(([k, v]) => (
                          <Mono key={k} size={12.5}>
                            {k}={v}
                          </Mono>
                        ))}
                        {row.has_image && <ImageIcon size={15} color={PALETTE.textMuted} alt="Has an image" />}
                      </Stack>
                    </TableCell>
                    <TableCell sx={{ maxWidth: 260 }} title={row.text}>
                      {row.text.slice(0, 60)}
                    </TableCell>
                    <TableCell>
                      {row.covered_by_rule ? (
                        <StateTag label={row.rule_reason ?? 'rule-covered'} tone="brand" size="small" />
                      ) : (
                        <StateTag label="needs judging" tone="warning" size="small" />
                      )}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={4} sx={{ p: 0, border: expanded === row.example_id ? undefined : 0 }}>
                      <Collapse in={expanded === row.example_id} timeout="auto" unmountOnExit>
                        <Stack spacing={1} sx={{ p: 2 }}>
                          <Button size="small" variant="outlined" onClick={() => void tryResolve(row)} sx={{ alignSelf: 'flex-start' }}>
                            Resolve this example
                          </Button>
                          {tryResults[row.example_id] &&
                            (typeof tryResults[row.example_id] === 'string' ? (
                              <Alert severity="error">{tryResults[row.example_id] as string}</Alert>
                            ) : (
                              <DecisionCard decision={tryResults[row.example_id] as Decision} />
                            ))}
                        </Stack>
                      </Collapse>
                    </TableCell>
                  </TableRow>
                </Fragment>
              ))}
            </TableBody>
          </Table>
        </Panel>
      </Stack>
    </>
  )
}
